from django.db import models
from django.urls import reverse
from django.contrib.auth.models import Permission, User
# Create your models here.
class Album(models.Model):
    user = models.ForeignKey(User,on_delete= models.CASCADE,default = 1)
    artist = models.CharField(max_length = 250)
    album_title = models.CharField(max_length = 250)
    genre = models.CharField(max_length = 100)
    album_logo = models.FileField()

    def get_absolute_url(self):
        return reverse('music:detail',kwargs={'pk':self.pk})
    
    def __str__(self):
        return self.album_title +'-'+ self.artist


class Song(models.Model):
    album = models.ForeignKey(Album, on_delete= models.CASCADE)
    file_type = models.CharField(max_length = 100)
    song_title = models.CharField(max_length = 250)
    audio_file = models.FileField(default = '')

    def get_absolute_url(self):
        return reverse('music:detail',kwargs={'pk':self.album.id})
    
    def __str__(self):
        return self.song_title