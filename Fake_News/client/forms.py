from django import forms

from client.models import RegisterModel



class RegisterForms(forms.ModelForm):
    class Meta:
        model=RegisterModel
        fields=("firstname","lastname","userid","password","phoneno","email","gender")