from django import forms
from .models import Pelaku

class PelakuForm(forms.ModelForm):
    class Meta:
        model = Pelaku
        fields = ['nama', 'umur', 'kasus', 'tim_dukung', 'foto_pelaku']
