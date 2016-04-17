
#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Dr. Igor Marfin'
SITENAME = u'Programmatic Statistics'
SITEURL = '"http://igormarfin.github.io/'

PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('Google+', 'https://plus.google.com/u/0/111481335270527541691/posts'),
         )

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
PLUGIN_PATHS = ['./plugins'] 
MARKUP = ('md', 'ipynb') 
PLUGINS = ['ipynb','share_post','tipue_search','tag_cloud'] 

# my additional settings to the pelican config
# mostly to support octoberpress theme
SITEURL = ''
THEME_NAME='octoberpress'
THEME_NAME='twitchy'
THEME_NAME='bootstrap3'



import os
import sys

sys.path.append(os.curdir)
from pelicanconf import *

path = os.path.curdir
THEME = '%s/theme' % path

FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'

# Set the article URL.
ARTICLE_URL = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'
YEAR_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/{date:%b}/index.html'

STATIC_PATHS=['images']

CACHE_CONTENT=True
LOAD_CONTENT_CACHE = True
#CACHE_PATH = 'cache'

#
THEME=THEME+"/"+THEME_NAME
if 'octoberpress' in THEME_NAME:
    GOOGLE_PLUS_ID = '111481335270527541691'
    GOOGLE_PLUS_ONE = True 
    SEARCH_BOX = True 
    
# support of the pelican-twitchy
if 'twitchy' in THEME_NAME:
    SITESUBTITLE=u""" "What we observe is not nature itself, but nature exposed to our method of questioning." -Werner Heisenberg """
    RECENT_POST_COUNT = 5 
    ##EXPAND_LATEST_ON_INDEX = True 
    OPEN_GRAPH = False 
    BOOTSTRAP_THEME = "readable" 
    PYGMENTS_STYLE = "autumn" 
    TYPOGRIFY = False
    ##SITELOGO="theme/images/preview_small.PNG"
    ##DISPLAY_RECENT_POSTS_ON_MENU = True
    #DISPLAY_PAGES_ON_MENU = True
    #DISPLAY_TAGS_ON_MENU  = True
    DISPLAY_CATEGORIES_ON_MENU = True
    CC_LICENSE =  "CC-BY-NC"
    SHARE = True
    DISQUS_NO_ID = True 


# 
# support of the bootstrap3
## new theme boot3
if 'bootstrap3' in THEME_NAME:
    BOOTSTRAP_THEME = 'yeti'
    PYGMENTS_STYLE = 'autumn'
    ABOUT_ME = """
    I'm a data scientist and programmer currently based in Chemnitz, Germany.
    <br> <br> 
    Previuosly, I was engaged with researches in particle and theoretical physics at CERN, Switzerland.
    <br> <br> 
    Now I am investigating and developing the techniques of the machine learning. 
    Also I am a big fan of the Bayesian Statistics. 
    <br> 
    <br>
    <b>Contact me:</b><br><br>  iggy.floyd.de at google.com
    """
    AVATAR = "http://www.googledrive.com/host/0B5OwgVT-YmdbZFBWSlRCWVdmVGM"

    # Blogroll
    LINKS =  (('Bitbucket', 'https://bitbucket.org/iggy_floyd/'),
          )
    #MENUITEMS = [('Photography','http://shots.saiwal.esy.es')]
    DEFAULT_PAGINATION = False
    DISPLAY_CATEGORIES_ON_MENU = False
    PAGE_EXCLUDES=['bootstrap.html', '404.html']
    DIRECT_TEMPLATES = ('index', 'categories', 'authors', 'archives', 'search')
    DISPLAY_ARTICLE_INFO_ON_INDEX = True
    DISPLAY_TAGS_ON_SIDEBAR=True
    DISPLAY_TAGS_INLINE=True
    DISPLAY_CATEGORIES_ON_MENU= True
    #BOOTSTRAP_NAVBAR_INVERSE=True
    #ARTICLE_PATHS = ['blog']
    ADDTHIS_PROFILE="ra-571208988724f2bf"
    ADDTHIS_FACEBOOK_LIKE = False   
    ADDTHIS_TWEET  = False
    CC_LICENSE = "CC-BY-NC-SA" 
    GOOGLE_PLUS_ID = '111481335270527541691'
    GOOGLE_PLUS_ONE = True 
   
# DISQUS SUPPORT
# DISQUS comments
DISQUS_SITENAME = "igormarfingithubio"    
