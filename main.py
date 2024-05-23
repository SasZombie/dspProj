import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import stft
from scipy.fft import fft, fftfreq
from scipy.io import wavfile

def generate_two_tones(volume):
    duration = 5
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    tone1 = volume * np.sin(2 * np.pi * 1500 * t)
    tone2 = volume * np.sin(2 * np.pi * 1550 * t)

    return tone1 + tone2

def make_non_linear(singal, gain = 0.01):
    return singal + gain + singal ** 5  

def record_sound(duration=1, sample_rate=44100):
    print("Started recording")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    print("Done recording")
    wavfile.write("output.wav", sample_rate, recording)
    
def main()->None:
    sample_rate = 44100  #Random value 
    duration = 1  
    frequency1 = 1500
    frequency2 = 1550  

    input_name = 'Razer Seiren V2 Pro:' #This should be changed on other systems!!

    sd.default.device = (input_name, None)
    #Asta este pentru timp
    # t1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    volume_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    ip3_values = []
    im3_values = []

    in_time = []
    frames = []
    i = 0
    
    tone = generate_two_tones(1)
    new_tone = np.int16(tone / np.max(np.abs(tone)) * 32767)
    wavfile.write("generated.wav", sample_rate, new_tone)
    

    for volume in volume_levels:

###########################################################
#if 0

        # tone_signal = generate_two_tones(volume)
        # tone_signal = make_non_linear(tone_signal)
#else
        input("Press enter to record!")
        record_sound(duration, sample_rate)
        _, tone_signal = wavfile.read('output.wav')
        tone_signal = tone_signal.ravel()
#endif
##########################################################

        """
        When two tones are present at the input of a nonlinear system, such as an amplifier, they generate intermodulation distortion products at the output. 
        For third-order intermodulation distortion, these products include sum and difference frequencies of the input tones, 
        as well as combinations of three times the input frequencies. In the STFT output, the IM3 power corresponds to the magnitude of the peak at the frequency 
        associated with the third-order intermodulation product. The magnitude represents the amplitude of the intermodulation distortion component at that frequency.
        """
        #Folosim audio in loc de tone sig
        f, _, Zxx = stft(tone_signal, fs=sample_rate, nperseg=2048, noverlap=512)
        
        fft_res = fft(tone_signal)

        freqs = fftfreq(len(fft_res), 1/sample_rate)
        mag = np.abs(fft_res)
        plt.plot(freqs, mag)
        
        filename = f'frame_{i}.png'
        frames.append(filename)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('FFT at volume: ' + str(volume * 100) + '%')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        i=i+1

        in_time.append([freqs, mag])
        
        f1_index = np.argmin(np.abs(f - frequency1))
        f2_index = np.argmin(np.abs(f - frequency2))
        im3_1_index = np.argmin(np.abs(f - (2 * frequency1 - frequency2)))
        im3_2_index = np.argmin(np.abs(f - (2 * frequency2 - frequency1)))
        
        fundamental_power1 = np.mean(np.abs(Zxx[f1_index]) ** 2)
        fundamental_power2 = np.mean(np.abs(Zxx[f2_index]) ** 2)
        im3_power1 = np.mean(np.abs(Zxx[im3_1_index]) ** 2)
        im3_power2 = np.mean(np.abs(Zxx[im3_2_index]) ** 2)

        im3_power = (im3_power1 + im3_power2) / 2

        ip3 = 10 * np.log10(3/2 * (fundamental_power1 + fundamental_power2) - im3_power / 2) 
        
        ip3_values.append(ip3)
        im3_values.append(im3_power)

       
        #This is chroma :3
        # plt.figure()
        # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        # plt.title(f'Spectrogram - Volume: {volume * 100}%')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.colorbar()
        # plt.show()
        
    nip3_values = np.array(ip3_values)
    nim3_values = np.array(im3_values)

    _, axes = plt.subplots(2, 3, figsize = (15, 10))

    j = 0
    for ith in range(2):
        for jth in range(3): 
            if ith == 1 and jth == 2:
                axes[ith, jth].plot(volume_levels, nip3_values, label='IP3')
                axes[ith, jth].plot(volume_levels, nim3_values, label='IM3')
                axes[ith, jth].set_xlabel('Volume Level (%)')
                axes[ith, jth].set_ylabel('Power (dB)')
                axes[ith, jth].set_title('IP3 and IM3 vs. Volume Level')
                axes[ith, jth].legend()
                axes[ith, jth].grid(True)
            else:
                axes[ith, jth].plot(in_time[j][0], in_time[j][1])
                axes[ith, jth].set_xlabel('Frequency (Hz)')
                axes[ith, jth].set_ylabel('Magnitude (dB)')
                axes[ith, jth].set_title('FFT at volume: ' + str(volume_levels[j] * 100) + '%')
                axes[ith, jth].grid(True)
                j+=1
        
      
    plt.show()

    plt.plot(volume_levels, nip3_values, label='IP3')
    plt.plot(volume_levels, nim3_values, label='IM3')
    plt.xlabel('Volume Level (%)')
    plt.ylabel('Power (dB)')
    plt.title('IP3 and IM3 vs. Volume Level')
    plt.legend()
    plt.grid(True)
    plt.show()


    with imageio.get_writer('plots.gif', mode='I', duration=0.01, loop = 0) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    for frame in frames:
        os.remove(frame)
if __name__ == "__main__":
    main()