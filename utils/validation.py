import tqdm
import torch
import torch.nn.functional as F


def validate(hp, args, generator, discriminator, valloader, dsp, writer, step, device):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (mel, audio) in enumerate(loader):
        mel = mel.to(device)
        audio = audio.to(device)
        noise = torch.randn(1, hp.gen.noise_dim, mel.size(2)).to(device)

        fake_audio = generator(mel, noise)[:,:,:audio.size(2)]

        mel_fake = dsp.wav_to_mel(fake_audio.squeeze(1))
        mel_real = dsp.wav_to_mel(audio.squeeze(1))

        mel_loss += F.l1_loss(mel_fake, mel_real).item()


    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    torch.backends.cudnn.benchmark = True
