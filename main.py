import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import wave
from scipy.signal import stft


def record_sound(duration=1, sample_rate=44100):
    print("Started recording")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    print("Done recording")
    audio_data = recording.flatten
    
    return recording

def main()->None:
    sample_rate = 44100  #Random value 
    duration = 1  
    frequency1 = 1500
    frequency2 = 1550  

    #Asta este pentru timp
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    volume_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    ip3_values = []
    im3_values = []

    for volume in volume_levels:

        
        # data = stream.read(CHUNK_SIZE)
                
        #unda de sinus 
        # tone1 = volume * np.sin(2 * np.pi * frequency1 * t)
        # tone2 = volume * np.sin(2 * np.pi * frequency2 * t)
        # tone_signal = tone1 + tone2
        # tone_signal = 2 * (tone2 - tone1)

        input("Press ENTER to start recording")
        tone_signal = record_sound(duration=2)
        print(tone_signal)

        # audio = np.frombuffer(data, dtype=np.int16)
        #Folosim audio in loc de tone sig


        """
        When two tones are present at the input of a nonlinear system, such as an amplifier, they generate intermodulation distortion products at the output. 
        For third-order intermodulation distortion, these products include sum and difference frequencies of the input tones, 
        as well as combinations of three times the input frequencies. In the STFT output, the IM3 power corresponds to the magnitude of the peak at the frequency 
        associated with the third-order intermodulation product. The magnitude represents the amplitude of the intermodulation distortion component at that frequency.
        """
        f, t, Zxx = stft(tone_signal, fs=sample_rate, nperseg=2048, noverlap=512)
        
        im3_peaks = np.where(np.abs(Zxx) == np.abs(Zxx).max())
        print(im3_peaks)
        f1_index = np.argmin(np.abs(f - frequency1))
        f2_index = np.argmin(np.abs(f - frequency2))
        im3_1_index = np.argmin(np.abs(f - (2 * frequency1 - frequency2)))
        im3_2_index = np.argmin(np.abs(f - (2 * frequency2 - frequency1)))
        
        fundamental_power1 = np.mean(np.abs(Zxx[f1_index]) ** 2)
        fundamental_power2 = np.mean(np.abs(Zxx[f2_index]) ** 2)
        im3_power1 = np.mean(np.abs(Zxx[im3_1_index]) ** 2)
        im3_power2 = np.mean(np.abs(Zxx[im3_2_index]) ** 2)

        # Calculate average IM3 power
        im3_power = (im3_power1 + im3_power2) / 2
        a = 1
        ip3 = 10 * np.log10(a * (fundamental_power1 + fundamental_power2) - im3_power / 2) 
        
        ip3_values.append(ip3[0])
        im3_values.append(im3_power[0])
        
        plt.figure()
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title(f'Spectrogram - Volume: {volume * 100}%')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()
        plt.show()
        
    print(ip3_values)
    print(im3_values)
    nip3_values = np.array(ip3_values)
    nim3_values = np.array(im3_values)

##################################
    print(nip3_values.shape)
    print(nim3_values.shape)
##################################

    plt.plot(volume_levels, nip3_values, label='IP3')
    plt.plot(volume_levels, nim3_values, label='IM3')
    plt.xlabel('Volume Level (%)')
    plt.ylabel('Power (dB)')
    plt.title('IP3 and IM3 vs. Volume Level')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()