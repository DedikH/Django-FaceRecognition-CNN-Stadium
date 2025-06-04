from django.db import models
from django import forms
from datetime import date

class Pelaku(models.Model):
    id = models.AutoField(primary_key=True)
    nama = models.CharField(max_length=256)
    umur = models.CharField(max_length=256)
    kasus = models.CharField(max_length=256)
    tim_dukung = models.CharField(max_length=256)
    foto_pelaku = models.ImageField(upload_to='foto_pelaku/', null=True, blank=True)
    tanggal_mulai = models.DateField(verbose_name="Ditetapkan Pada", null=True, blank=True)
    tanggal_berakhir = models.DateField(verbose_name="Berlaku Hingga", null=True, blank=True)

    @property
    def lama_hukuman(self):
        if self.tanggal_mulai and self.tanggal_berakhir:
            return (self.tanggal_berakhir - self.tanggal_mulai).days
        return "-"


    def __str__(self):
        return self.nama

