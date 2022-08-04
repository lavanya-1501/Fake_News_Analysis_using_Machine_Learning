from django import forms

from management.models import AdminModel


class AdminForm(forms.ModelForm):
    class Meta:
        model = AdminModel
        fields = ('newsid','title','text','label')