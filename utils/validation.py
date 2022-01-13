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

        mel_fake = dsp.wav_to_mel(fake_audio.squeeze(1).cpu().numpy())
        mel_real = dsp.wav_to_mel(audio.squeeze(1).cpu().numpy())

        mel_fake = torch.tensor(mel_fake).unsqueeze(0)
        mel_real = torch.tensor(mel_real).unsqueeze(0)

        mel_loss += F.l1_loss(mel_fake, mel_real).item()

        if idx < hp.log.num_audio:
            audio = audio[0][0].cpu().detach().numpy()
            fake_audio = fake_audio[0][0].cpu().detach().numpy()
            writer.log_fig_audio(audio, fake_audio, None, None, idx, step)

    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    torch.backends.cudnn.benchmark = True
