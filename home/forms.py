from django import forms
from home.models import Image
class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = ('image',)
        labels = {
            'image': 'Please select an image file',
        }