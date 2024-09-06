import numpy as np
import os
import time
import cv2
import glob
import pickle
import copy
import queue
from threading import Thread
import multiprocessing as mp
import torch
from tritonclient.utils import triton_to_np_dtype

from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal
from tqdm import tqdm
import tritonclient.grpc.aio as grpcclient
import tritonclient.utils.shared_memory as shm

from webrtc import PlayerStreamTrack


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def __mirror_index(size, index):
    # size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


def inference(render_event, batch_size, face_imgs_path, audio_feat_queue, audio_out_queue, res_frame_queue,
              triton_url):
    loop = asyncio.get_event_loop()

    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print('start inference')

    # wav2lip256外部推理Triton服务的客户端
    triton_client = grpcclient.InferenceServerClient(url=triton_url)
    shm_ip_handle, inputs, shm_op_handle, outputs = None, None, None, None

    async def prepare_shm(triton_client: grpcclient.InferenceServerClient, batch_size):
        """为Triton推理准备共享内存"""
        # Create shared memory region for output and store shared memory handle
        outputs_byte_size = 12582912
        shm_op_handle = shm.create_shared_memory_region("output_data",
                                                        "/output_simple",
                                                        outputs_byte_size)

        # Create shared memory region for input and store shared memory handle
        inputs_byte_size = 25247744
        shm_ip_handle = shm.create_shared_memory_region(
            "input_data", "/input_simple", inputs_byte_size)
        # Register shared memory region for outputs with Triton Server| Register shared memory region for inputs with Triton Server
        await asyncio.gather(
            triton_client.register_system_shared_memory("output_data", "/output_simple", outputs_byte_size),
            triton_client.register_system_shared_memory(
                "input_data", "/input_simple", inputs_byte_size)
        )
        # Set the parameters to use data from shared memory
        _input0_byte_size = 81920
        _input1_byte_size = 25165824
        outputs_byte_size = 12582912
        inputs = []
        inputs.append(grpcclient.InferInput("input0", [batch_size, 1, 80, 16], "FP32"))
        inputs[-1].set_shared_memory("input_data", _input0_byte_size)
        inputs.append(grpcclient.InferInput("input1", [batch_size, 6, 256, 256], "FP32"))
        inputs[-1].set_shared_memory("input_data", _input1_byte_size, offset=_input0_byte_size)

        outputs = [grpcclient.InferRequestedOutput("output")]
        outputs[-1].set_shared_memory("output_data", outputs_byte_size)

        return shm_ip_handle, inputs, shm_op_handle, outputs

    try:
        shm_ip_handle, inputs, shm_op_handle, outputs = loop.run_until_complete(prepare_shm(triton_client, batch_size))
        while True:
            if render_event.is_set():
                try:
                    mel_batch = audio_feat_queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue

                is_all_silence = True
                audio_frames = []
                for _ in range(batch_size * 2):  # 一个视频帧对应2个音频帧
                    frame, type = audio_out_queue.get()  # frame:shape(320,) dtype float32
                    audio_frames.append((frame, type))
                    if type == 0:
                        is_all_silence = False

                if is_all_silence:
                    for i in range(batch_size):
                        res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                        index = index + 1
                        # print(f"is_all_silence: {audio_frames[i * 2:i * 2 + 2]=}")
                else:
                    # print('infer=======')
                    t = time.perf_counter()
                    img_batch = []
                    for i in range(batch_size):
                        idx = __mirror_index(length, index + i)
                        face = face_list_cycle[idx]
                        img_batch.append(face)
                    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                    img_masked = img_batch.copy()
                    img_masked[:, face.shape[0] // 2:] = 0

                    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                    mel_batch = np.transpose(mel_batch, (0, 3, 1, 2)).astype(np.float32)
                    img_batch = np.transpose(img_batch, (0, 3, 1, 2)).astype(np.float32)

                    # 远程Triton推理
                    loop.run_until_complete(
                        inf_through_triton(triton_client, mel_batch, img_batch, shm_ip_handle, inputs, outputs, ))

                    pred: np.ndarray = shm.get_contents_as_numpy(shm_op_handle,
                                                                 triton_to_np_dtype('FP32'),
                                                                 [batch_size, 3, 256, 256])
                    pred = pred.transpose(0, 2, 3, 1) * 255.
                    # pred: (16,256,256,3)
                    counttime += (time.perf_counter() - t)
                    count += batch_size
                    # _totalframe += 1
                    if count >= 100:
                        print(f"------actual avg infer fps:{count / counttime:.4f}")
                        count = 0
                        counttime = 0
                    for i, res_frame in enumerate(pred):
                        # self.__pushmedia(res_frame,loop,audio_track,video_track)
                        res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                        # print(f"not is_all_silence: {audio_frames[i * 2:i * 2 + 2]=}")
                        index = index + 1
                    # print('total batch time:',time.perf_counter()-starttime)
            else:
                time.sleep(1)
    finally:
        async def shutdown_triton_client(triton_client: grpcclient.InferenceServerClient, shm_ip_handle, shm_op_handle):
            """优雅退出Triton客户端"""
            try:
                await asyncio.gather(triton_client.unregister_system_shared_memory("input_data"),
                                     triton_client.unregister_system_shared_memory("output_data"))
            except:
                ...
            try:
                shm.destroy_shared_memory_region(shm_ip_handle)
            except:
                ...
            try:
                shm.destroy_shared_memory_region(shm_op_handle)
            except:
                ...
            assert len(shm.mapped_shared_memory_regions()) == 0

        loop.run_until_complete(shutdown_triton_client(triton_client, shm_ip_handle, shm_op_handle))


async def inf_through_triton(triton_client: grpcclient.InferenceServerClient,
                             mel_batch: np.ndarray,
                             img_batch: np.ndarray,
                             shm_ip_handle,
                             inputs,
                             outputs):
    """
    通过外部wav2lip256的Triton服务进行推理
    Returns:

    """
    # 模型
    model_name = "wav2lip256"
    model_version = "1"
    # Put input data values into shared memory
    _input0_byte_size = 81920
    shm.set_shared_memory_region(shm_ip_handle, [mel_batch])
    shm.set_shared_memory_region(shm_ip_handle, [img_batch], offset=_input0_byte_size)
    # 启动推理
    output = await triton_client.infer(model_name=model_name, model_version=model_version, inputs=inputs,
                                       outputs=outputs)
    # output = output.get_output("output")
    # return output


@torch.no_grad()
class wav2lip256TritonReal(BaseReal):
    def __init__(self, opt):
        super().__init__(opt)
        # self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps  # 20 ms per frame

        #### musetalk
        self.avatar_id = opt.avatar_id
        self.avatar_path = f"./data/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.face_imgs_path = f"{self.avatar_path}/face_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        # self.__loadmodels()
        self.__loadavatar()

        self.asr = LipASR(opt, self)
        self.asr.warm_up()
        # self.__warm_up()

        self.render_event = mp.Event()
        mp.Process(target=inference, args=(self.render_event, self.batch_size, self.face_imgs_path,
                                           self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue,
                                           opt.triton_url)).start()

    def __loadavatar(self):
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)

    def put_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):  # 16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track: PlayerStreamTrack | None = None):

        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:  # 全为静音数据，只需要取fullimg
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None:  # 有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                    # if not self.custom_opt[audiotype].loop and self.custom_index[audiotype]>=len(self.custom_img_cycle[audiotype]):
                    #     self.curr_state = 1  #当前视频不循环播放，切换到静音状态
                else:
                    combine_frame = self.frame_list_cycle[idx]
            else:
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                # combine_frame = get_image(ori_frame,res_frame,bbox)
                # t=time.perf_counter()
                combine_frame[y1:y2, x1:x2] = res_frame
                # print('blending time:',time.perf_counter()-t)

            image = combine_frame  # (outputs['image'] * 255).astype(np.uint8) shape:(h ,w, 3) dtype:np.uint8
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

            for audio_frame in audio_frames:
                frame, type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
        print('musereal process_frames thread stop')

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        # if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        self.render_event.set()  # start infer process render
        _starttime = time.perf_counter()
        # _totalframe=0
        while not quit_event.is_set():
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
            if video_track._queue.qsize() >= 5:
                print('sleep qsize=', video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        self.render_event.clear()  # end infer process render
        print('musereal thread stop')
