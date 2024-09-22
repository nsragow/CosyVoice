# Debian GNU/Linux 12 (bookworm)
import torch

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")


from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

sample_rate = 44100
print("loading model")
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
# prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)

# cross_lingual usage
# prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
#     torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], 22050)

    # cross_lingual usage
print("Starting load wav")
prompt_speech_16k = load_wav('NoahMonolougingcopy.wav', sample_rate)
print("starting result")
result = cosyvoice.inference_cross_lingual('<|ja|>こんにちは、今日は何をしていますか', prompt_speech_16k, stream=False)
print("finished result")
for i, j in enumerate(result):
    wav_file = 'cross_lingual_{}.wav'.format(i)
    print(f'saving {wav_file}')
    torchaudio.save(wav_file, j['tts_speech'], 22050)
# for i, j in enumerate(cosyvoice.inference_cross_lingual('<|zh|>你好，你今天在做什么', prompt_speech_16k, stream=False)):
#     torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], 22050)