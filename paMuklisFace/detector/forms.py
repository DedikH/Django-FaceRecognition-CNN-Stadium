from django import forms
from .models import Pelaku

class PelakuForm(forms.ModelForm):
    class Meta:
        model = Pelaku
        fields = [
            'nama', 'umur', 'kasus', 'tim_dukung', 'foto_pelaku', 'tanggal_mulai', 'tanggal_berakhir'
        ]
        widgets = {
            'tanggal_mulai': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'tanggal_berakhir': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        }
