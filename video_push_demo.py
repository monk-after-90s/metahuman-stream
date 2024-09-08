import asyncio
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from aiortc.contrib.media import MediaPlayer
import uvloop
import signal

# 使用uvloop作为事件循环策略
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

pc = None
video_track = None
audio_track = None


async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


class FileVideoStreamTrack(VideoStreamTrack):
    def __init__(self, player: MediaPlayer):
        super().__init__()  # don't forget this!
        self.player = player

    async def recv(self):
        frame = await self.player.video.recv()
        if frame:
            return frame
        else:
            return None


class FileAudioStreamTrack(AudioStreamTrack):
    def __init__(self, player: MediaPlayer):
        super().__init__()  # don't forget this!
        self.player = player

    async def recv(self):
        frame = await self.player.audio.recv()
        if frame:
            return frame
        else:
            return None


async def get_pc(push_url):
    global pc
    global video_track
    global audio_track

    pc = RTCPeerConnection()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()

    # 创建MediaPlayer实例读取本地视频文件
    player = MediaPlayer('video.mp4', format='mp4', options={'rtsp_transport': 'udp'})

    video_track = FileVideoStreamTrack(player)
    audio_track = FileAudioStreamTrack(player)

    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    # 设置本地描述
    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


async def gracefully_close():
    try:
        await pc.close()
    except:
        pass
    try:
        not video_track or video_track.stop()
    except:
        pass
    try:
        not audio_track or audio_track.stop()
    except:
        pass
    loop.stop()


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
