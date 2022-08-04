"""Fake_News URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls.static import static
from django.conf.urls import url
from django.contrib import admin
from client import views as user_view
from Fake_News import settings

urlpatterns = [
    url(r'^admin/', admin.site.urls),

    url(r'^$', user_view.login, name="login"),
    url(r'^register/$', user_view.register, name="register"),
    url(r'^mydetails/$', user_view.mydetails, name="mydetails"),
    url(r'^upload_news/$', user_view.upload_news, name="upload_news"),
    url(r'^upload_dataset/$', user_view.upload_dataset, name="upload_dataset"),
    url(r'^view_upload/$', user_view.view_upload, name="view_upload"),
    url(r'^analysis/$', user_view.analysis, name="analysis"),

    url(r'^topreal/$', user_view.topreal, name="topreal"),
    url(r'^topfake/$', user_view.topfake, name="topfake"),
    url(r'^count/$', user_view.count, name="count"),
    url(r'^tfidf/$', user_view.tfidf, name="tfidf"),
    url(r'^hashing/$', user_view.hashing, name="hashing"),
    url(r'^passiveaggressive/$', user_view.passiveaggressive, name="passiveaggressive"),


]+static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)
