A post with an Image
####################

:date: 2016-2-7 21:50
:tags: blog,post,image
:category: BuildingThisBlog
:slug: a-post-with-an-image

Sometimes you would like to post articles with images. What you can do is the following

.. code-block:: bash

   make post_with_image

   # to test the html output
   make html
   make serve

   # to publish to github.com
   make publish


`make post_with_image` command. Basicaly, it performs


.. code-block:: bash

   make add_entry_rst A post with an Image,blog\\\,post\\\,image,BuildingThisBlog
   cat how-to-post-with-image.rst >> $(BLOG_SOURCE)/content/*a-post-with-an-image.rst
   mkdir -p $(BLOG_SOURCE)/content/images
   cp pelican-plugin-post-stats-medium-example.png $(BLOG_SOURCE)/content/images/
   echo "

   Here is my image

    .. figure:: /images/pelican-plugin-post-stats-medium-example.png
             :align: right

             This is the caption of the figure.

   " >> $(BLOG_SOURCE)/content/*a-post-with-an-image.rst


Also it is important that your `pelicanconf.py` has `images` in the static path:


.. code-block:: bash

   STATIC_PATHS=['images']
 


Here is my image

.. figure:: /images/pelican-plugin-post-stats-medium-example.png
    :align: right

    This is the caption of the figure.



That's it. 

