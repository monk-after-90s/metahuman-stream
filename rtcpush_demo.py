import asyncio
from typing import Union
from PIL import Image
import numpy as np
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, VideoStreamTrack, AudioStreamTrack
from av import Packet
from av.frame import Frame
from av import VideoFrame
import fractions

pc = None
ast = None
pst = None
audio_tracker = None
video_tracker = None


async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


class PlayerStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.frame_count = 0

        img = Image.open('girl.jpeg')
        try:
            img_array = np.array(img)
        finally:
            img.close()

        new_frame = VideoFrame.from_ndarray(img_array, format="rgb24")
        new_frame.time_base = fractions.Fraction(1, 25)
        self.new_frame = new_frame

    async def recv(self) -> Union[Frame, Packet]:
        self.frame_count += 1
        self.new_frame.pts = self.frame_count
        return self.new_frame


async def get_pc(push_url):
    global pc
    global ast
    global pst
    global audio_tracker
    global video_tracker

    pc = RTCPeerConnection()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()

    ast = AudioStreamTrack()
    audio_tracker = pc.addTrack(ast)

    pst = PlayerStreamTrack()
    video_tracker = pc.addTrack(pst)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(get_pc('http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream'))
    try:
        loop.run_forever()
    finally:
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
                not audio_tracker or await audio_tracker.stop()
            except:
                pass
            try:
                not video_tracker or await video_tracker.stop()
            except:
                pass


        loop.run_until_complete(gracefully_close())  # 如果生产环境不行，则只能用signal
        loop.stop()
        loop.close()
        print('=============================\nBye!')
