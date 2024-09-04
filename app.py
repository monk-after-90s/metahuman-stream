# server.py
import json
from threading import Thread, Event
import multiprocessing
from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from webrtc import HumanPlayer
import argparse
import asyncio

nerfreals = []
statreals = []


async def llm_response_generator(message):
    from llm.LLM import LLM
    if opt.llm_type == "ChatGPT":
        llm = LLM().init_model(opt.llm_type, model_path=opt.llm_model, api_key=opt.api_key,
                               proxy_url=opt.proxy_url, openai_base_url=opt.base_url)
    else:
        raise NotImplementedError
    # llm = LLM().init_model('Gemini', model_path= 'gemini-pro',api_key='Your API Key', proxy_url=None)
    # llm = LLM().init_model('ChatGPT', model_path='gpt-3.5-turbo', api_key='Your API Key')
    # llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
    sentence = ""
    async for token in llm.chat_stream(message):
        # 拆句
        for c in token:
            if c in '“”‘’"。,，;；:：、？?！!' and sentence:
                yield sentence
                print(f"{sentence=}")
                sentence = ""
            else:
                sentence += c


#####webrtc###############################
pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


async def run(push_url):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[0])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    # 事件循环初始化
    loop = asyncio.get_event_loop()
    # asyncio.set_event_loop(loop)

    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')

    parser.add_argument('--num_rays', type=int, default=4096 * 16,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1,
                        help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8,
                        help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0,
                        help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01,
                        help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2,
                        help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='',
                        help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000,
                        help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true',
                        help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    # parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')  #
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)  # todo 适配，比如24。根据日志，这个也可能不需要动。
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    # musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)  # 慎动

    # parser.add_argument('--customvideo', action='store_true', help="custom video")
    # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
    # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts model，可选edgetts、xtts、gpt-sovits、vits-simple")  # xtts gpt-sovits vits-simple
    parser.add_argument('--voicename', type=str, default='zh-CN-YunjianNeural',
                        help="指定的发声人，比如--tts edgetts时可选项查看https://github.com/jsbxyyx/tts_java；--tts vits-simple时这个参数无效")  # xtts gpt-sovits
    parser.add_argument('--vits_simple_url', type=str,
                        default='https://vits_simple.excn.top/voice/bert-vits2_n?speaker=Dr_yang&text={}',
                        help="--tts vits-simple时的语音合成地址，要求里面有一对花括号，用以占位具体合成语音的文本。默认值https://vits_simple.excn.top/voice/bert-vits2_n?speaker=Dr_yang&text={}")

    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')  # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='ernerf')  # musetalk wav2lip
    # model为wav2lip256tritonreal时依赖的参数
    parser.add_argument(
        "--triton_url",
        type=str,
        required=False,
        default="localhost:8001",
        help="model==wav2lip256tritonreal时的Triton推理服务。具体部署参考项目digitalAvatar中介绍如何部署该Triton服务",
    )

    parser.add_argument('--transport', type=str, default='rtcpush')  # rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')  # rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  # multi session count
    parser.add_argument('--listenport', type=int, default=8010)

    # LLM
    parser.add_argument('--llm_type', type=str, required=False, default='ChatGPT',
                        help="大模型类型。这里的对话只能处理单条用户问题，无历史记录，生产环境还是需要单独处理大模型请求流程。")
    parser.add_argument('--llm_model', type=str, required=False, default='gpt-4o-mini', help="大模型名字")
    parser.add_argument('--api_key', type=str, required=False, default='', help="大模型API私钥")
    parser.add_argument('--proxy_url', type=str, required=False, default='', help="大模型API代理地址")
    parser.add_argument('--base_url', type=str, required=False,
                        default='https://api.openai.com/v1',
                        help="大模型API的BaseURL。对于ChatGPT规范的API，这个值需要以“/v1”结尾")

    opt = parser.parse_args()
    # app.config.from_object(opt)
    # print(app.config)
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    if opt.model == 'ernerf':
        from ernerf.nerf_triplane.provider import NeRFDataset_Test
        from ernerf.nerf_triplane.utils import *
        from ernerf.nerf_triplane.network import NeRFNetwork
        from nerfreal import NeRFReal

        # assert test mode
        opt.test = True
        opt.test_train = False
        # opt.train_camera =True
        # explicit smoothing
        opt.smooth_path = True
        opt.smooth_lips = True

        assert opt.pose != '', 'Must provide a pose source'

        # if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.exp_eye = True
        opt.smooth_eye = True

        if opt.torso_imgs == '':  # no img,use model output
            opt.torso = True

        # assert opt.cuda_ray, "Only support CUDA ray mode."
        opt.asr = True

        if opt.patch_size > 1:
            # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
            assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."
        seed_everything(opt.seed)
        print(opt)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(opt)

        criterion = torch.nn.MSELoss(reduction='none')
        metrics = []  # use no metric in GUI for faster initialization...
        print(model)
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset_Test(opt, device=device).dataloader()
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        # we still need test_loader to provide audio features for testing.
        for _ in range(opt.max_session):
            nerfreal = NeRFReal(opt, trainer, test_loader)
            nerfreals.append(nerfreal)
    elif opt.model == 'musetalk':
        from musereal import MuseReal

        print(opt)
        for _ in range(opt.max_session):
            nerfreal = MuseReal(opt)
            nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal

        print(opt)
        for _ in range(opt.max_session):
            nerfreal = LipReal(opt)
            nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip256tritonreal':
        from wav2lip256tritonreal import wav2lip256TritonReal

        print(opt)
        for _ in range(opt.max_session):
            nerfreal = wav2lip256TritonReal(opt)
            nerfreals.append(nerfreal)
    for _ in range(opt.max_session):
        statreals.append(0)

    if opt.transport == 'rtmp':
        thread_quit = Event()
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_static('/', path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp':
        pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    print('start http server; http://<serverip>:' + str(opt.listenport) + '/' + pagename)


    def run_server(runner):
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport == 'rtcpush':
            loop.run_until_complete(run(opt.push_url))
        loop.run_forever()


    run_server(web.AppRunner(appasync))
