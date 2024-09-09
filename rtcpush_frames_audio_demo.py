import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from typing import Union
from PIL import Image
import numpy as np
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, AudioStreamTrack, \
    RTCRtpSender
from av import Packet
from av.frame import Frame
from av import AudioFrame, VideoFrame
import fractions
from fractions import Fraction
import signal

pc = None
ast: None | AudioStreamTrack = None
pst = None
audio_sender: None | RTCRtpSender = None
video_sender = None


async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, kind="video"):
        super().__init__()
        self.frame_count = 0
        self.kind = kind

        img = Image.open('girl.jpeg')
        try:
            img_array = np.array(img)
        finally:
            img.close()

        new_frame = VideoFrame.from_ndarray(img_array, format="rgb24")
        new_frame.time_base = fractions.Fraction(1, 25)  # 视频25fps
        self.new_frame = new_frame

    async def recv(self) -> Union[Frame, Packet]:
        if self.kind == "video":
            self.frame_count += 1
            self.new_frame.pts = self.frame_count
            return self.new_frame
        else:
            return get_audio_frame()


async def get_pc(push_url):
    global pc
    global ast
    global pst
    global audio_sender
    global video_sender

    pc = RTCPeerConnection()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()

    ast = PlayerStreamTrack(kind="audio")
    audio_sender = pc.addTrack(ast)

    pst = PlayerStreamTrack()
    video_sender = pc.addTrack(pst)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


async def gracefully_close():
    try:
        await pc.close()
    except:
        pass
    try:
        not ast or ast.stop()
    except:
        ...
    try:
        not pst or pst.stop()
    except:
        pass
    try:
        not audio_sender or await audio_sender.stop()
    except:
        pass
    try:
        not video_sender or await video_sender.stop()
    except:
        pass
    loop.stop()


pts = 0
time_base = Fraction(1, 16000)


def get_audio_frame():
    frame = np.zeros(320, dtype=np.float32)
    frame = (frame * 32767).astype(np.int16)
    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
    new_frame.planes[0].update(frame.tobytes())
    new_frame.sample_rate = 16000

    global pts
    new_frame.pts = pts
    pts += 320  # 16000/50，音频50fps
    new_frame.time_base = time_base

    return new_frame


def safely_exit():
    asyncio.create_task(gracefully_close())


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, safely_exit)
    loop.add_signal_handler(signal.SIGINT, safely_exit)

    loop.create_task(get_pc('http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream'))

    try:
        loop.run_forever()
    finally:
        loop.close()
    exit()
