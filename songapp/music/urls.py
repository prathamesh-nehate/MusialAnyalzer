from django.conf.urls import url
from . import views
from django.urls import path,include

app_name = 'music' #namespace

urlpatterns = [
    path('<pk>/',views.DetailView.as_view(),name = 'detail'),
    path('user/register/',views.UserFormView.as_view(),name = 'register'),
    path('user/login/',views.login_user,name = 'login'),
    path('user/logout/',views.logout_user,name = 'logout'),
    path('album/add/',views.create_album,name = 'album-add'),
    path('albumupdate/<pk>/',views.AlbumUpdate.as_view(),name = 'album-update'),
    path('album/<pk>/delete',views.AlbumDelete.as_view(),name = 'album-delete'),
    path('album/<album_id>/addsong',views.create_song,name = 'song-add'),
    path('songs/show/',views.SongView.as_view(),name = 'songs'),
    path('<song_id>',views.genre,name = 'genre'),
    path('chart/spectrogram/<song_id>/',views.spectogram,name = 'spectrogram'),
    path('chart/harmonic/<song_id>/',views.harmonic,name = 'harmonic'),
    path('chart/mfccs/<song_id>/',views.mfccs,name = 'mfccs'),
    path('chart/decompose/<song_id>/',views.decomp,name = 'decomp'),
    path('songs/compare/<song_id>/',views.compare,name = 'compare'),
    path('songs/cspectrogram/<song_id>/',views.compare_spectrogram,name = 'cspectrogram'),
    path('songs/cmfccs/<song_id>/',views.compare_mfccs,name = 'cmfccs'),
    path('',views.index, name = 'index'),
]