from __future__ import print_function
from django.shortcuts import render,get_object_or_404,redirect
from django.http import HttpResponse,Http404,request
from django.template import loader
from django.views.generic.edit import CreateView,UpdateView,DeleteView
from django.views import generic
from django.views.generic import View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login,logout
from django.db.models import Q
from .models import Album,Song
from .forms import UserForm,SongForm,AlbumForm
from django.contrib import messages

import matplotlib
matplotlib.use("Agg") #backend to raster png 
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sweetify


'''
def index(request):
    all_albums = Album.objects.all()
    #template = loader.get_template('music/index.html')
    context = {
        'all_albums' : all_albums,
    }
    return render(request, 'music/index.html', context )
def detail(request,album_id):
    
    album = get_object_or_404(Album,pk = album_id)
    return render(request, 'music/detail.html', {'album' : album} )
'''
AUDIO_FILE_TYPES = ['wav', 'mp3', 'ogg', 'au']
IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']

class IndexView(generic.ListView):
    template_name = 'music/index.html'
    context_object_name = 'all_albums'

    def get_queryset(self):
        return Album.objects.all()

class SongView(generic.ListView):
    template_name = 'music/song.html'
    context_object_name = 'all_songs'

    def get_queryset(self):
        return Song.objects.all()

class DetailView(generic.DetailView):
    model = Album
    template_name = 'music/detail.html'
    
class AlbumCreate(CreateView):
    model = Album
    fields = [
        'artist',
        'album_title',
        'genre',
        'album_logo',
    ]

class AlbumUpdate(UpdateView):
    model = Album
    fields = [
        'artist',
        'album_title',
        'genre',
        'album_logo',
    ]

class AlbumDelete(DeleteView):
    model = Album
    success_url = reverse_lazy('music:index')

class SongCreate(CreateView):
    model = Song
    fields =[
        'album',
        'file_type',
        'song_title',
        'audio_file',
    ]

class SongDelete(DeleteView):
    model = Song
    success_url = reverse_lazy('music:details')

class UserFormView(View):
    form_class = UserForm
    template_name = 'music/register.html'

    #display blank form
    def get(self,request):
        form = self.form_class(None)
        return render(request,self.template_name,{'form':form})

    #process form data
    def post(self,request):
        form = self.form_class(request.POST)

        if form.is_valid():
            user = form.save(commit=False)
            #normalize data
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user.set_password(password)
            user.save()
            user = authenticate(username = username,password = password)
            if user is not None:
                #check for revoked accounts
                if user.is_active:
                    login(request,user)
                    return redirect('music:index')

        return render(request,self.template_name,{'form' : form})

class SongCreateView(View):
    form_class = SongForm
    template_name = 'music/song_form.html'
    def get(self,request,album_id):
        form = self.form_class(None)
        return render(request,self.template_name,{'form':form})

    def post(self,request,album_id):
        form = self.form_class(request.POST)

        album = get_object_or_404(Album, pk=album_id)
        if form.is_valid():
            albums_songs = album.song_set.all()
            for s in albums_songs:
                if s.song_title == form.cleaned_data.get("song_title"):
                    context = {
                        'album': album,
                        'form': form,
                        'error_message': 'You already added that song',
                    }
                    return render(request,self.template_name, context)
            song = form.save(commit=False)
            song.album = album
            song.audio_file = request.request['audio_file']
            file_type = song.audio_file.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in AUDIO_FILE_TYPES:
                context = {
                    'album': album,
                    'form': form,
                    'error_message': 'Audio file must be WAV, MP3, or OGG',
                }
                return render(request, self.template_name, context)

            song.save()
            return render(request, 'music/detail.html', {'album': album})
        context = {
            'album': album,
            'form': form,
        }
        return render(request,self.template_name, context)

def create_song(request, album_id):
    form = SongForm(request.POST or None, request.FILES or None)
    album = get_object_or_404(Album, pk=album_id)
    if form.is_valid():
        albums_songs = album.song_set.all()
        for s in albums_songs:
            if s.song_title == form.cleaned_data.get("song_title"):
                context = {
                    'album': album,
                    'form': form,
                    'error_message': 'You already added that song',
                }
                return render(request, 'music/song_form.html', context)
        song = form.save(commit=False)
        song.album = album
        song.audio_file = request.FILES['audio_file']
        file_type = song.audio_file.url.split('.')[-1]
        file_type = file_type.lower()
        if file_type not in AUDIO_FILE_TYPES:
            context = {
                'album': album,
                'form': form,
                'error_message': 'Audio file must be WAV, MP3, or OGG',
            }
            return render(request, 'music/song_form.html', context)

        song.save()
        return render(request, 'music/detail.html', {'album': album})
    context = {
        'album': album,
        'form': form,
    }
    return render(request, 'music/song_form.html', context)

def create_album(request):
    if not request.user.is_authenticated:
        return render(request, 'music/login.html')
    else:
        form = AlbumForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            album = form.save(commit=False)
            album.user = request.user
            album.album_logo = request.FILES['album_logo']
            file_type = album.album_logo.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in IMAGE_FILE_TYPES:
                context = {
                    'album': album,
                    'form': form,
                    'error_message': 'Image file must be PNG, JPG, or JPEG',
                }
                return render(request, 'music/create_album.html', context)
            album.save()
            return render(request, 'music/detail.html', {'album': album})
        context = {
            "form": form,
        }
        return render(request, 'music/create_album.html', context)


def index(request):
    if not request.user.is_authenticated:
        return render(request, 'music/login.html')
    else:
        albums = Album.objects.filter(user=request.user)
        song_results = Song.objects.filter()
        query = request.GET.get("q")
        if query:
            albums = albums.filter(
                Q(album_title__icontains=query) |
                Q(artist__icontains=query)
            ).distinct()
            song_results = song_results.filter(
                Q(song_title__icontains=query)
            ).distinct()
            return render(request, 'music/index.html', {
                'all_albums': albums,
                'songs': song_results,
            })
        else:
            return render(request, 'music/index.html', {'all_albums': albums})
def logout_user(request):
    logout(request)
    form = UserForm(request.POST or None)
    context = {
        "form": form,
    }
    return render(request, 'music/login.html', context)


def login_user(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                albums = Album.objects.filter(user=request.user)
                return render(request, 'music/index.html', {'all_albums': albums})
            else:
                return render(request, 'music/login.html', {'error_message': 'Your account has been disabled'})
        else:
            return render(request, 'music/login.html', {'error_message': 'Invalid login'})
    return render(request, 'music/login.html')

def spectogram(request,song_id):
    import matplotlib.pyplot as plt
    import librosa.display

    import numpy as np
    import librosa


    song1 = get_object_or_404(Song,pk = song_id)
    songname = song1.audio_file.path
    #print(songname)
    y, sr = librosa.load(songname,duration=10)
    y = y[:100000] # shorten audio a bit for speed

    window_size = 1024
    window = np.hanning(window_size)
    stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)

    # For plotting headlessly
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/spectrogram.html', {'data' : data} )

def harmonic(request,song_id):
    import numpy as np
    import matplotlib.pyplot as plt

    import librosa
    import librosa.display

    #Fetch the clip
    song1 = get_object_or_404(Song,pk = song_id)
    songname = song1.audio_file.path

    #Load the clip 
    y, sr = librosa.load(songname, offset=40, duration=10)
    #y = y[y!=0]

    # Compute the short-time Fourier transform of y
    D = librosa.stft(y)

    # Decompose D into harmonic and percussive components
    #D = D{harmonic} + D{percussive}
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    # Pre-compute a global reference power from the input spectrum
    rp = np.max(np.abs(D))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
    plt.colorbar()
    plt.title('Full spectrogram')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), y_axis='log')
    plt.colorbar()
    plt.title('Harmonic spectrogram')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Percussive spectrogram')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/harmonic.html', {'data' : data} )

def mfccs(request,song_id):
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    #import soundfile as sf
    import librosa.display

    song1 = get_object_or_404(Song,pk = song_id)
    songname = song1.audio_file.path
    y,sr = librosa.load(songname,duration=5)
    mfccs = librosa.feature.mfcc(y=y,sr=sr)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(mfccs,x_axis='time',y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/mfccs.html', {'data' : data} )

def decomp(request,song_id):
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    #import soundfile as sf
    import librosa.display
    song1 = get_object_or_404(Song,pk = song_id)
    songname = song1.audio_file.path
    #print(songname)
    y, sr = librosa.load(songname,duration=10)
    S = np.abs(librosa.stft(y))
    comps, acts = librosa.decompose.decompose(S, n_components=16,sort = True)
    
    
    plt.figure(figsize=(10,8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='log', x_axis='time')
    plt.title('Input spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(3, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(comps,ref=np.max),y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Components')
    plt.subplot(3, 2, 4)
    librosa.display.specshow(acts, x_axis='time')
    plt.ylabel('Components')
    plt.title('Activations')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    S_approx = comps.dot(acts)
    librosa.display.specshow(librosa.amplitude_to_db(S_approx,ref=np.max),y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig("harmonic.png")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/decomp.html', {'data' : data} )
def pl(request,song_id):    
    import librosa
    import numpy as np 
    import matplotlib.pyplot as plt 
    import librosa.display
    import matplotlib.animation as animation
    import time 
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_webagg import FigureCanvasAgg as FigureCanvas
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    x = 0
    y,sr = librosa.load("media\State_Of_Grace_dPJ5DQF.mp3",duration=10)
    z = []
    while x <= y.shape[0]:
        z.append(y[x:x+2205])
        x += 2205
    print(len(z))
    t = time.time()
    for i in range(len(z)):
        librosa.display.waveplot(z[i],sr=sr)
        plt.pause(0.01)       
    plt.clf()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/chart.html', {'data' : data} )

temp = 0
def compare(request,song_id):
    songs = Song.objects.all()
    global temp 
    temp = song_id
    return render(request,'music/compare.html',{'all_songs' : songs })

def compare_harmonic(request,song_id):
    import numpy as np
    import matplotlib.pyplot as plt

    import librosa
    import librosa.display

    def harm_perc(song_id):
        song1 = get_object_or_404(Song,pk = song_id)
        songname = song1.audio_file.path

        #Load the clip 
        y, sr = librosa.load(songname, offset=40, duration=10)

        # Compute the short-time Fourier transform of y
        D = librosa.stft(y)

        # Decompose D into harmonic and percussive components
        #D = D{harmonic} + D{percussive}
        D_harmonic, D_percussive = librosa.decompose.hpss(D)
       
        # Pre-compute a global reference power from the input spectrum
        rp = np.max(np.abs(D))
        return D,D_harmonic,D_percussive,rp

    D,D_harmonic,D_percussive,rp = harm_perc(song_id)
    D1,D_harmonic1,D_percussive1,rp1 = harm_perc(temp)
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
    plt.colorbar()
    plt.title(get_object_or_404(Song,pk = song_id).song_title + '\n\n Full spectrogram')

    plt.subplot(3, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), y_axis='log')
    plt.colorbar()
    plt.title('Harmonic spectrogram')

    plt.subplot(3, 2, 6)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Percussive spectrogram')
    
    plt.subplot(3, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1), ref=rp1), y_axis='log')
    plt.colorbar()
    plt.title(get_object_or_404(Song,pk = temp).song_title + '\n\n Full spectrogram')

    plt.subplot(3, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic1), ref=rp1), y_axis='log')
    plt.colorbar()
    plt.title('Harmonic spectrogram')

    plt.subplot(3, 2, 5)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive1), ref=rp1), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Percussive spectrogram')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/harmonic_compare.html', {'data' : data} )


def compare_mfccs(request,song_id):
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    def mfcc(song_id):
        song1 = get_object_or_404(Song,pk = song_id)
        songname = song1.audio_file.path
        y,sr = librosa.load(songname,duration=5)
        mfccs = librosa.feature.mfcc(y=y,sr=sr)
        return mfccs

    mfccs1 = mfcc(song_id)
    mfccs2 = mfcc(temp)
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.title(get_object_or_404(Song,pk = song_id).song_title)
    librosa.display.specshow(mfccs1,x_axis='time',y_axis='log')
    plt.subplot(122)
    plt.title(get_object_or_404(Song,pk = temp).song_title)
    librosa.display.specshow(mfccs2,x_axis='time',y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/mfccs_compare.html', {'data' : data} )

def compare_spectrogram(request,song_id):
    import matplotlib.pyplot as plt
    import librosa.display

    import numpy as np
    import librosa

    def out(song_id):
        song1 = get_object_or_404(Song,pk = song_id)
        songname = song1.audio_file.path
        #print(songname)
        y, sr = librosa.load(songname,duration=10)
        y = y[:100000] # shorten audio a bit for speed

        window_size = 1024
        window = np.hanning(window_size)
        stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        return out

    out1 = out(song_id)
    out2 = out(temp)
    # For plotting headlessly
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    #plt.Figure(figsize=(6,8))
    plt.subplot(211,)
    plt.title(get_object_or_404(Song,pk = song_id).song_title)
    p = librosa.display.specshow(librosa.amplitude_to_db(out1, ref=np.max), y_axis='log',x_axis = 'time')
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(212)
    plt.title(get_object_or_404(Song,pk = temp).song_title)
    p = librosa.display.specshow(librosa.amplitude_to_db(out2, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return render(request, 'music/spectrogram.html', {'data' : data} )


def genre(request,song_id):    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
    import csv
    import pandas as pd
    import numpy as np
    data = pd.read_csv(r"G:\Programs\SDL\S\songapp\music\data.csv")
    data = data.drop(['filename'],axis=1)


    genres = data.iloc[:,-1]
    #print(genres)
    encode = LabelEncoder()
    y = encode.fit_transform(genres)
    scaler = MinMaxScaler()
    #print(y)
    z = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    #print(z)
    X = np.array(data.iloc[:, :-1])
    #X = scaler.fit_transform(np.array(data.iloc[:, :-1],dtype=float))
    #X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #print('::::::::::::',X_test.shape)
    '''
    from keras import models
    from keras import layers
    x_val = X_train[:200]
    partial_x_train = X_train[200:]

    y_val = y_train[:200]
    partial_y_train = y_train[200:]

    model = models.Sequential()

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(partial_x_train,
            partial_y_train,
            epochs=100,
            batch_size=512,
            validation_data=(x_val, y_val))
    results = model.evaluate(X_test, y_test)
    '''
    from sklearn import svm
    cf = svm.SVC(probability=True,gamma='auto')
    cf.fit(X_train,y_train)
    #print(cf.score(X_test,y_test))
    import librosa
    import numpy as np

    #y,sr = librosa.load("hth.mp3",duration=30)
    song1 = get_object_or_404(Song,pk = song_id)
    songname = song1.audio_file.path
    #print(songname)
    y, sr = librosa.load(songname,duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y,sr=sr)
    tonne = librosa.feature.tonnetz(y=y,sr=sr)
    oenv = librosa.onset.onset_strength(y=y,sr=sr)
    rmse = librosa.feature.rms(y=y)
    harmonics = librosa.effects.harmonic(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y,sr=sr)
    spec_band = librosa.feature.spectral_bandwidth(y=y,sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mfccs = librosa.feature.mfcc(y=y,sr=sr)
    calc = np.array([np.mean(chroma_stft),np.mean(tonne),np.mean(oenv),np.mean(harmonics),np.mean(rmse),np.mean(spec_cent),np.mean(spec_band),np.mean(rolloff),np.mean(zcr)])
    for i in mfccs:
        calc = np.append(calc,np.mean(i))
    #print(calc.shape)

    val = cf.predict(calc.reshape(1,29))[0]
    predicted = z[val]

    #sweetify.sweetalert(request, 'Westworld is awesome')
    print('Predicted:  ',z[val])
    
    messages.success(request,predicted)
    album = get_object_or_404(Album,pk = song1.album.id)
    context = {
        'album' : album,
    }
    return render(request,'music/detail.html',context)
