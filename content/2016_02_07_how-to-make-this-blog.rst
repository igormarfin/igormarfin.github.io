HOW-TO make this blog
#####################

:date: 2016-2-7 13:41
:tags: create pelican blog
:category: BuildingThisBlog
:slug: how-to-make-this-blog






These are only few  steps which you can do, if you want to have such blog:

.. code-block:: bash

   git clone https://iggy_floyd@bitbucket.org/iggy_floyd/blog_with_ipython_notebook.git
   cd blog_with_ipython_notebook
   make 
   
Then if ``make`` command is successful, you can initialize your blog and add first posts:


.. code-block:: bash

  make blog_ini
  make GitHubPages_ini
  make add_entry_rst About Me,about me,AboutMe
  echo >> src/content/*about-me.rst
  echo >> src/content/*about-me.rst
  echo "Hi! My name is Igor Marfin. I am going to start my blog..." >> src/content/*about-me.rst
  make add_entry_ipython `pwd`/backtesting-strategy-with-kalman-model.ipynb,Kalman Modeling,Ipython,Ipython

  make html
  make serve

  make publish


That's it. More information can be found in the ``docs``:


.. code-block:: bash

   firefox doc/README.html

   # or as an alternative way to get docs
   cat README.wiki
