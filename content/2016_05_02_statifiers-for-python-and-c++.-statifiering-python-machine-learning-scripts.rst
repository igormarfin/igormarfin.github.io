Statifiers for Python and C++. Statifiering python Machine-Learning scripts
###########################################################################

:date: 2016-5-2 20:32
:tags: statifier,GO,XGO,Pyinstaller,CGO,Cython,Python,C++
:category: Programming
:slug: statifiers-for-python-and-c++.-statifiering-python-machine-learning-scripts

==============================================================================
 Statifiers for Python and C++. Statifiering python Machine-Learning scripts
==============================================================================





:Author: Igor Marfin
:Contact: igor.marfin@unister.de
:Organization: private
:Date: Apr 26, 2016, 9:06:51 AM
:Status: draft
:Version: 1
:Copyright: This document has been placed in the public domain. You
            may do with it as you wish. You may copy, modify,
            redistribute, reattribute, sell, buy, rent, lease,
            destroy, or improve it, quote it at length, excerpt,
            incorporate, collate, fold, staple, or mutilate it, or do
            anything else to it that your or anyone else's heart
            desires.

.. admonition:: Dedication

    For statisticians & programmers

.. admonition:: Abstract

    This projects reveals a few secrets about the ways how to produce cross-platform programs in Python and C++.

.. meta::
   :keywords: pandas,ipython,dataframe,trading, statistics, finance
   :description lang=en:  Statifiering Python and C++





.. contents:: Table of Contents





----------------------------------------------------------------
Introduction to the topic and motivation
----------------------------------------------------------------

I want to introduce you to my motivation which forces me to develop this project.

You may be wondering, "why would you want to have statically linked python applications?". 
There are a few reasons why dynamic linking for Python and C/
C++ programs is superior :


* Fixes (either security or only bug) have to be applied to only one place: the new DSO(s). 
  If various applications are linked statically, all of them would have to be relinked. 
  By the time the problem is discovered the sysadmin usually forgot which apps are built with the problematic library. 

* Security measures like load address randomization cannot be used. 
  With statically linked applications, only the stack and heap address can be randomized. 
  All text has a fixed address in all invocations. 
  With dynamically linked applications, the kernel has the ability to load all DSOs at arbitrary addresses, 
  independent from each other. 
  In case the application is built as a position independent executable (PIE) even this code can be 
  loaded at random addresses. 
  Fixed addresses (or even only fixed offsets) are the dreams of attackers. 
  And no, it is not possible in general to generate PIEs with static linking. 


* More efficient use of physical memory. All processes share the same physical pages for the code in the DSOs. 
  With prelinking startup times for dynamically linked code is as good as that of statically linked code.


* All kinds of features in the libc (locale (through iconv), NSS, IDN, ...) 
  require dynamic linking to load the appropriate external code. 
  We have very limited support for doing this in statically linked code. 
  But it requires that the dynamically loaded modules available at runtime must come from the same glibc 
  version as the code linked into the application. 
  And it is completely unsupported to dynamically load DSOs this way which are not part of glibc. 


* Related, trivial NSS modules can be used from statically linked apps directly. 
  If they require extensive dependencies (like the LDAP NSS module, not part of glibc proper) 
  this will likely not work. 

* No accidental violation of the (L)GPL. Should a program which is statically linked be given to a third party, 
  it is necessary to provide the possibility to regenerate the program code.


* Tools and hacks like ltrace, LD_PRELOAD, LD_PROFILE, LD_AUDIT don't work at the statical linking.
  These can be effective debugging and profiling, especially for remote debugging where 
  the user cannot be trusted with doing complex debugging work.


After all, "static linking is evil".  But, how would we answer questions, pointed out below,
if the dynamic linking were only possible at the modern Operating Systems:

* Were it possible to achieve cross platform compatibility in C/C++ applications?

* Were it possible to run Python programs on other machines without requiring that they have installed Python?

* Were it possible to run Python programs on other machines without requiring that they have had
  the same versions of the same libraries installed that you do?

* Were it possible to distribute the python code under the commercial licence? The ``.pyc`` files 
  can be easily decompiled (disassembled) and  the code, that checks the license file, can be removed.
  It is not desirable that the commercial code could be read by customers, 
  because the code might be stolen or at least the "novel ideas". Also for 
  proprietary or security-conscious applications, this is unacceptable.

.. role:: red

.. raw:: html

    <style> .red {color:red; font-size: 200%;} </style>


It would be ":red:`NOT`, :red:`NOT` and once more :red:`NOT`" possible.


In this tutorial, I will try to review technologies helping to build cross-platform (cross-compiled)
solutions in Python and C/C++. My `project`__, which I call a ``Python Virtual Machine``, is aimed to
illustrate considered approaches and techniques. 


.. __: https://bitbucket.org/iggy_floyd/virtual_machine_python




.. figure:: /images/Crossplatform.jpg
   :align: center
   :width: 770px
   :height: 320px
   :scale: 100
   :alt: Cross-platform compatibility

   Cross-platform applications

---------------------------------
Protecting a Python codebase
---------------------------------

There are numerous discussions and arguments about protection of the python codebase with 
exploting the bytecode `ref1`__, `ref2`__, `ref3`__.

.. __: http://bits.citrusbyte.com/protecting-a-python-codebase/ 

.. __: https://github.com/citrusbyte/python-obfuscation 

.. __: http://stackoverflow.com/questions/261638/how-do-i-protect-python-code

The generated ``*.pyc`` files are quite simple binaries containing ``a magic number (four bytes)``,
``a timestamp (four bytes)``, ``a code object (marshalled code)``. 
It’s fair to think that this is a secure mechanism. 
It’s not hard to apply the ``dis`` module to get a notion of what the code is about, 
and there are tools which can be used to translate the op-codes back to human-friendly strings, for example 
that’s already done by the ``uncompyle2 package``. 

Usually at forums, when someone raises a request on help, like 


>>> I'm building a Python application and don't want to force my clients to install Python and modules. I also want to make my application closed-source.


three solutions are proposed:

* `pyinstaller`__

* `nuitka`__

* `cython`__

.. __ : https://mborgerson.com/creating-an-executable-from-a-python-script

.. __ : http://nuitka.net/doc/user-manual.html#command-line

.. __ : https://pypi.python.org/pypi/Cython/


I have tested all of them. 
Unfortunately I have to admit that ``nuitka`` has failed to produce C++ wrapping code from python scripts. 
Therefore I restrict myself to consideration of ``pyinstaller``'s  and ``cython``'s ways to ``binarize`` python code. 
Before we proceed further, I spend some time on installation instructions.





----------------------------------
Installation of the tutorial
----------------------------------

.. Important::
  
    The project relies on the presence of ``Autotools`` in the system.
    Before, proceed further, please, install them: ``sudo apt-get install autotools-dev``.




Simply, clone the project 

.. code-block:: bash

    git clone https://iggy_floyd@bitbucket.org/iggy_floyd/virtual_machine_python.git

and test a configuration of your system that you have components like
    
* python 

* scientific python modules

* golang and its supporting tools

installed. It is done via the ``make``  command as it's  shown below

.. code-block:: bash
    
    make


-------------------------------------------------------
Go: a swiss army knife for cross-platform applications
-------------------------------------------------------



A few years ago I tinkered with Go, a language created and promoted by google. I found it cool. I have developed a few small projects, `go-cpp`__,
`go-tutorials`__ and `test-otto (a Go's javascript interpeter)`__ in Go. 

There is a number of reasons that folks are in love with golang. One the most mentioned is the static linking.

As long as the source being compiled is native go, the go compiler will statically link the executable. 
Though when you need to use cgo, then the compiler has to use its external linker. 

.. __ : https://bitbucket.org/iggy_floyd/go-cpp

.. __ : https://bitbucket.org/iggy_floyd/go-tutorial

.. __ : https://bitbucket.org/iggy_floyd/test-otto


.. Important::

    At least Go compiler and linker are needed to be installed in the system. Please, read and follow installation instructions found at 
    the official `web site`__ of the Golang.  Also it would be helpful to install the ``godocdown`` `tool`__ generating Go documentation in the 
    Markdown format:

    .. code-block:: bash

       go get github.com/robertkrimen/godocdown/godocdown


.. __ : https://golang.org/doc/install

.. __ : https://github.com/robertkrimen/godocdown


----------------------------------------------------------------    
How to create a binary from a Python Script with ``PyInstaller``
----------------------------------------------------------------    

.. Important::

    I would like to mention that all experiments with statifiers have been done in the 32-bit ``wheezy``-Debian VM.

    .. code-block:: bash

       lsb_release -a; uname -a

    >>> Distributor ID:	Debian
    >>> Description:	Debian GNU/Linux 7.8 (wheezy)
    >>> Release:	7.8
    >>> Codename:	wheezy
    >>> Linux debian 3.2.0-4-686-pae #1 SMP Debian 3.2.65-1+deb7u2 i686 GNU/Linux


In our first exercise we experiment with the ``Pyinstaller``.  There is a `nice post`__ from Matt Borgerson
concerning this topic. Basic ideas and a few examples were taken  from this publication to illustrate the technology.
One can look at the examples of the  ``pyinstaller`` application by reading the corresponded ``HOW-TO``:

.. code-block:: bash

   cat pyinstaller/HOW-TO


.. the text below can be obtained running a command ``./rsTCodeBlockProducer.sh bash pyinstaller/HOW-TO``

.. code-block:: bash 

   # installation of the wxPython
   sudo apt-get install python-wxgtk2.8
   
   # installation of the Pyinstaller
   sudo pip install pyinstaller
   
   # test 1: a windowed application
   pyinstaller --onefile --windowed window_app.py
   
   # test 2: a simple script utilizing numpy and pandas 
   # -d for debugging purpose
   pyinstaller --onefile -d numpy_pandas_test.py
   


The fist appication ``window_app.py`` is a simple ``Hello World`` windowed application built on top of the cross-platform ``wxWidgets``
`C++ library`__  helping to create appications  for Windows, Mac OS X, Linux and other platforms with a single code base. 


.. __ : https://mborgerson.com/creating-an-executable-from-a-python-script

.. __ : http://www.wxpython.org/


The code of the python script is shown below.


.. the text below can be obtained running a command ``./rsTCodeBlockProducer.sh python pyinstaller/window_app.py``


.. code-block:: python 

   #!/usr/bin/env python
   import wx
   app = wx.App(False)
   frame = wx.Frame(None, wx.ID_ANY, "Hello World")
   frame.Show(True)
   app.MainLoop()


We can easily compile it to the binary 

.. code-block:: bash

   cd pyinstaller
   pyinstaller --onefile --windowed window_app.py


and test the obtained file.


.. code-block:: bash

   # you can start a window by a command
   #./dist/window_app

   ldd dist/window_app


>>>	linux-gate.so.1 =>  (0xb776b000)
>>>	libdl.so.2 => /lib/i386-linux-gnu/i686/cmov/libdl.so.2 (0xb774d000)
>>>	libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb7734000)
>>>	libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb75cf000)
>>>	/lib/ld-linux.so.2 (0xb776c000)


The next example demostrates building a binary from a python script intened to be used in 
some data analysis. This script utilizes pandas and numpy libraries. To run the ``pyinstaller`` on the script properly, you need to install
``libgdk-pixbuf2.0-dev``, `ref_about_proble`__.


.. __ : http://stackoverflow.com/questions/35020183/pyinstaller-not-compiling-large-program-in-python2-7-64-bit-ubuntu


.. code-block:: bash

   sudo apt-get install libgdk-pixbuf2.0-dev


.. obtained with ``./rsTCodeBlockProducer.sh python pyinstaller/numpy_pandas_test.py``

.. required a few hacks: add "src_root_path_or_glob = src_root_path_or_glob.split('\n')[-1]" @  line 421 of /usr/local/lib/python2.7/dist-packages/PyInstaller/building/utils.py
   add  
    try:
     import re
     m=re.search('\{.*\}',txt,re.DOTALL)
     txt = m.group(0)
    except:
      return ''

    @ line 127 of  /usr/local/lib/python2.7/dist-packages/PyInstaller/utils/hooks/__init__.py

Afterwards, the script

.. code-block:: python 

   
   #!/usr/bin/env python


   # Main Program starts here
   def main():
    import sys
    sys.path = ['/usr/local/lib/python2.7/dist-packages'] + sys.path # to fix the problem with numpy: this replaces  1.6 version by 1.9
 
    print sys.path
    import numpy as np
    # it doesn't work because of the incompatibility with numpy installed in my VM
    #import pandas as pd

    # 1) print a numpy array
    print np.array([1,2,3,4])

    # it doesn't work because of the incompatibility with numpy installed in my VM
    # 2) take a look at the dataset
    #df = pd.read_csv("train.csv")
    #print '**'*60,'\n'
    #print df.head()
    #print '**'*60,'\n'

   main()

can be compiled and run.


.. code-block:: bash

   cd pyinstaller
   pyinstaller --onefile numpy_pandas_test.py
   dist/numpy_pandas_test
   ldd dist/numpy_pandas_test   
   
>>>    ['/usr/local/lib/python2.7/dist-packages', '/tmp/_MEItpKh8l']
>>>    [1 2 3 4]

>>>    linux-gate.so.1 =>  (0xb77a5000)
>>>    libdl.so.2 => /lib/i386-linux-gnu/i686/cmov/libdl.so.2 (0xb7785000)
>>>    libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb776c000)
>>>    libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb7607000)
>>>    /lib/ld-linux.so.2 (0xb77a6000)


If you test this binary on a machine with other version of the Linux, say,
for example, Ubuntu with 64-bit architecture,

.. code-block:: bash

   lsb_release -a
   uname -m


>>>    No LSB modules are available.
>>>    Distributor ID:	Ubuntu
>>>    Description:	Ubuntu 14.04.3 LTS
>>>    Release:	14.04
>>>    Codename:	trusty

>>>    x86_64


you will probably get either an error of the missing `32-bit dynamic loader`__ ``ld-linux.so.2``  like


>>> ./numpy_pandas_test :error “No such file or directory”


.. __ : http://stackoverflow.com/questions/2716702/no-such-file-or-directory-error-when-executing-a-binary


or an error of the missing dynamic libraries like ``libz.so.1``.


>>> ./numpy_pandas_test: error while loading shared libraries: libz.so.1: cannot open shared object file: No such file or directory

The first problem can be solved by installing the 32-bit loader and the 32-bit Standard C library.

.. code-block:: bash

   sudo apt-get install libc6-i386 lib32stdc++6 lib32gcc1 lib32ncurses5 lib32z1 


The second one reqiures more sophisticated solution. I provide it via the scripts ``build_package.sh`` and ``statifier.py``.


.. code-block:: bash
   
   ls pyinstaller/{build_package,statifier}*


>>> pyinstaller/build_package.sh  pyinstaller/statifier.py


You simply build the package with all need dynamic libraries inside,


.. code-block:: bash

   cd  pyinstaller
   pyinstaller --onefile numpy_pandas_test.py
   ln -s dist/numpy_pandas_test numpy_pandas_test; ./build_package.sh numpy_pandas_test; rm numpy_pandas_test
   
then you copy it to the remote machine (VM, Docker Container etc) with a different OS


.. code-block:: bash

   scp vagrant@172.17.0.1:/home/vagrant/project/package_numpy_pandas_test.tgz 


and execute it here.



.. code-block:: bash

   tar -xf package_numpy_pandas_test.tgz 
   cd package_numpy_pandas_test/
   ./numpy_pandas_test


>>>    ['/usr/local/lib/python2.7/dist-packages', '/tmp/_MEItpKh8l']
>>>    [1 2 3 4]



To close the topic about ``pyinstaller``, I want to add that  ``pyinstaller`` makes packaging of all python modules and 
extenstions ``(*.so files found in the python modules)``,  which are being "imported" in the script, and the python runtime environment 
in one executable file or directory.


-----------------------------------------------
How to compile python scripts with ``Cython``
-----------------------------------------------

As the `Cython wiki`__ suggests,

>>>   Typically Cython is used to create extension modules for use from Python programs. 
>>>   It is, however, possible to write a standalone programs in Cython. 
>>>   This is done via embedding the Python interpreter with the --embed option.

.. __ : https://github.com/cython/cython/wiki/EmbeddingCython

we can build C programs from python scripts. To prove this approach, we have to install ``Cython`` first.

I refer you to  the instructions on installation of the ``Cython`` found in the file ``cython/HOW-TO``.

.. obtained with ``./rsTCodeBlockProducer.sh bash cython/HOW-TO``

.. code-block:: bash 

   # How to install Cython
   # https://pypi.python.org/pypi/Cython/
   
   # method 1
   # pip install Cython --install-option="--no-cython-compile"
   
   # method 2, https://github.com/cython/cython/wiki/Installing
   wget http://cython.org/release/Cython-0.24.tar.gz
   tar -xf Cython-0.24.tar.gz 
   cd Cython-0.24/
   sudo python setup.py install
   
   # to build a binary from a python script
   make 

   
If ``Cython`` has been successfully installed, you can look at the python script which we are going to translate to a C standalone program.

.. obtained with ``./rsTCodeBlockProducer.sh python cython/titanic_logistic_regression_grid_search_roc_plot_example.py``

.. code-block:: python 

   #!/usr/bin/env python
   
   ''' This is an example of the application of the Logistic fucntion to the dataset analysis. 
       Logistic function is helpful to model binary models
   
       This is a analysis with dedicated sklearn sub-modules.
   
       The basic idea is taken from 
       http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#example-linear-model-plot-iris-logistic-py
       http://scikit-learn.org/stable/auto_examples/grid_search_digits.html	
       http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/
   
       DataFrameImputer was taken from 
       http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
   
       About Precision and Recall
       http://en.wikipedia.org/wiki/Precision_and_recall
   
       About Type 1 and Type 2 errors 
       http://en.wikipedia.org/wiki/Type_I_and_type_II_errors
   
       About scorring parameters for sklearn classifiers
       http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
   
       About plottig ROC in sklearn
       http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html
   
       About plottig Precision vs Recall in sklearn
       http://scikit-learn.org/stable/auto_examples/plot_precision_recall.html
   
   '''
   
   import sys
   sys.path = ['/usr/local/lib/python2.7/dist-packages'] + sys.path # to fix the problem with numpy: this replaces  1.6 version by 1.9
   import pandas as pd
   from sklearn.ensemble import ExtraTreesClassifier
   from sklearn.cross_validation import cross_val_score
   from sklearn.preprocessing import Imputer
   from sklearn import preprocessing
   import numpy as np
   from sklearn.base import TransformerMixin
   from StringIO import StringIO
   import prettytable    
   import statsmodels.api as sm
   import pylab as pl 
   from sklearn import linear_model
   from sklearn.cross_validation import train_test_split
   from sklearn.grid_search import GridSearchCV
   from sklearn.metrics import classification_report
   from sklearn.metrics import roc_curve, auc,precision_recall_curve
   
   class DataFrameImputer(TransformerMixin):
   
       def __init__(self):
           """Impute missing values.
   
           Columns of dtype object are imputed with the most frequent value 
           in column.
   
           Columns of other types are imputed with mean of column.
   
           """
       def fit(self, X, y=None):
   
           self.fill = pd.Series([X[c].value_counts().index[0]
               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
               index=X.columns)
   
           return self
   
       def transform(self, X, y=None):
           return X.fillna(self.fill)
   
   
   
   def transform_data(train_df,input_variable_names):
    ''' transform categorial data in train_df to numerical'''
   
   
    input_variable_types = map(lambda x: 'string' if 'object' in train_df[x].ftypes else train_df[x].ftypes,input_variable_names )
    labelencoders = []
   
   
    for i,x in enumerate(input_variable_types):
   ## here transform string label to numeric labels
     if 'string' in x:
      values =  map(lambda x: x if x!=np.nan else 'NaN' ,train_df[input_variable_names[i]].get_values())
      labelencoders+=[preprocessing.LabelEncoder()]   
      labelencoders[-1].fit(values)
      train_df[input_variable_names[i]] =  train_df[input_variable_names[i]].apply(lambda x: labelencoders[-1].transform(x))
   
    return labelencoders 
   
   
   # 1) get data for an analysis
   df = pd.read_csv("train.csv")
   
   # 2) take a look at the dataset
   print '**'*60,'\n'
   print df.head()
   print '**'*60,'\n'
   
   # 3) summarize data
   print '**'*60,'\n'
   print df.describe()
   print '**'*60,'\n'
   
   # 6) we are going to analyze a regressiong factors between 'Survied' as a target and Pclass,Sex,Age,Fare,Embarked
   # also we introduce dummy categories for all of the Pclass,Sex,Embarked  to do this analysis.
   input_variable_names = ['Pclass','Sex','Embarked','Age','Fare']
   
   # we need to understand the type of variables
   input_variable_types = map(lambda x: 'string' if 'object' in df[x].ftypes else df[x].ftypes,input_variable_names )
   
   # do variable transformation
   encoders = transform_data(df,input_variable_names)
   
   # apply  DataFrameImputer to fix np.nan
   df = DataFrameImputer().fit(df).transform(df)
   
   cols_to_keep = ['Survived','Pclass','Sex','Embarked','Age','Fare']
   data = df[cols_to_keep]
   
   
   print '**'*60,'\n'
   print data.head()
   print '**'*60,'\n'
   
   # 7) split the sample for train and test sub-samples
   # Split the dataset in two equal parts
   train_cols = ['Pclass','Sex','Embarked','Age','Fare']
   X_train, X_test, y_train, y_test = train_test_split(data[train_cols],data['Survived'] , test_size=0.5, random_state=0)
   
   # Set the parameters by cross-validation and scores
   tuned_parameters = {'C': [1, 10, 100, 10000,1e5,1e6]}
   # the whole list of scoring types can be found here:
   # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
   
   # choose the best 'recall' strategy and start the grid
   scoring = 'recall'
   clf = GridSearchCV(linear_model.LogisticRegression(C=1e5), tuned_parameters, n_jobs = -1,verbose=1 ,cv=5, scoring=scoring)
   clf.fit(X_train, y_train)
   print("Best parameters:")
   bestParams = clf.best_estimator_.get_params()
   for p in sorted(tuned_parameters.keys()):
     print "\t %s: %f" % (p, bestParams[p])
   
   print("Detailed classification report:")
   print()
   print("The model is trained on the full development set.")
   print("The scores are computed on the full evaluation set.")
   print()
   y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)
   print(classification_report(y_true, y_pred))
   print()
   
   
   # 9) Compute ROC curve and area under the curve
   probas_ = clf.best_estimator_.predict_proba(X_test)
   fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
   roc_auc = auc(fpr, tpr)
   pl.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
   pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
   pl.xlim([-0.05, 1.05])
   pl.ylim([-0.05, 1.05])
   pl.xlabel('False Positive Rate (FPR or Type-1 error )')
   pl.ylabel('True Positive Rate (TPR or recall or Type-2 error)')
   pl.title('Receiver operating characteristic example')
   pl.legend(loc="lower right")
   pl.show()
   
   # 10) Compute Precision (PositivePredictiveValue=TP/(TP+FP))  vs recall (TPR=P/(TP+FN))
   # http://scikit-learn.org/stable/auto_examples/plot_precision_recall.html
   ppv, tpr, thresholds = precision_recall_curve(y_test, probas_[:, 1])
   roc_auc = auc(tpr, ppv)
   pl.plot(tpr, ppv, lw=1, label='Precsion-Recall curve (area = %0.2f)' % (roc_auc))
   pl.xlim([-0.05, 1.05])
   pl.ylim([-0.05, 1.05])
   pl.ylabel('PPV (or precision)')
   pl.xlabel('True Positive Rate (TPR or recall)')
   pl.title('Precsion-Recall Curve example')
   pl.legend(loc="lower right")
   pl.show()
   
   
   raw_input("Press the Enter key...")
   

This is a simple example of the Logistic Regression analysis, which I wrote a few years ago during my experiments with the `scikit-learn package`__  to
solve the problem given at the `Kaggle Competition 2012`__, **Titanic: Machine Learning from Disaster**. The script does nothing except training 
the Logistic regressor with optimized parameters and calculating ROC, Precision-Recall curves for obtained model.


.. __ : http://scikit-learn.org/stable/



.. __ : https://www.kaggle.com/c/titanic


Let's build the executable binary, ``embedded`` from the script ``titanic_logistic_regression_grid_search_roc_plot_example.py`` and run it.


.. code-block:: bash


   cd cython
   make
   ./embedded



The result of both command is shown below.


.. code-block:: bash  

   gcc -pthread -c titanic_logistic_regression_grid_search_roc_plot_example.c -I/usr/include/python2.7 -I/usr/include/python2.7 
   gcc -pthread -o embedded titanic_logistic_regression_grid_search_roc_plot_example.o -L/usr/lib -L/usr/lib/python2.7/config -lpython2.7 -lpthread -ldl  -lutil -lm -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions

   
   ************************************************************************************************************************ 
   
      PassengerId  Survived  Pclass  \
   0            1         0       3   
   1            2         1       1   
   2            3         1       3   
   3            4         1       1   
   4            5         0       3   
   
                                                   Name     Sex  Age  SibSp  \
   0                            Braund, Mr. Owen Harris    male   22      1   
   1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
   2                             Heikkinen, Miss. Laina  female   26      0   
   3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1   
   4                           Allen, Mr. William Henry    male   35      0   
   
      Parch            Ticket     Fare Cabin Embarked  
   0      0         A/5 21171   7.2500   NaN        S  
   1      0          PC 17599  71.2833   C85        C  
   2      0  STON/O2. 3101282   7.9250   NaN        S  
   3      0            113803  53.1000  C123        S  
   4      0            373450   8.0500   NaN        S  
   ************************************************************************************************************************ 
   
   ************************************************************************************************************************ 
   
          PassengerId    Survived      Pclass         Age       SibSp  \
   count   891.000000  891.000000  891.000000  714.000000  891.000000   
   mean    446.000000    0.383838    2.308642   29.699118    0.523008   
   std     257.353842    0.486592    0.836071   14.526497    1.102743   
   min       1.000000    0.000000    1.000000    0.420000    0.000000   
   25%     223.500000    0.000000    2.000000   20.125000    0.000000   
   50%     446.000000    0.000000    3.000000   28.000000    0.000000   
   75%     668.500000    1.000000    3.000000   38.000000    1.000000   
   max     891.000000    1.000000    3.000000   80.000000    8.000000   
   
               Parch        Fare  
   count  891.000000  891.000000  
   mean     0.381594   32.204208  
   std      0.806057   49.693429  
   min      0.000000    0.000000  
   25%      0.000000    7.910400  
   50%      0.000000   14.454200  
   75%      0.000000   31.000000  
   max      6.000000  512.329200  
   ************************************************************************************************************************ 
   
   ************************************************************************************************************************ 
   
      Survived  Pclass  Sex  Embarked  Age     Fare
   0         0       3    1         2   22   7.2500
   1         1       1    0         0   38  71.2833
   2         1       3    0         2   26   7.9250
   3         1       1    0         2   35  53.1000
   4         0       3    1         2   35   8.0500
   ************************************************************************************************************************ 
   
   Fitting 5 folds for each of 6 candidates, totalling 30 fits
   [Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    0.0s
   [Parallel(n_jobs=-1)]: Done  28 out of  30 | elapsed:    0.2s remaining:    0.0s
   [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    0.2s finished

   Best parameters:
   	 C: 10.000000
   Detailed classification report:
   ()
   The model is trained on the full development set.
   The scores are computed on the full evaluation set.
   ()
                precision    recall  f1-score   support
   
             0       0.84      0.82      0.83       283
             1       0.69      0.72      0.71       163
   
   avg / total       0.78      0.78      0.78       446
   
   ()
   Press the Enter key...

The figures below illustrate ROC and Rrecision-Recall curve of the model predicting survival probability  in the ``Titanic-Disaster`` problem


.. figure:: /images/figure_1.png
   :align: center
   :width: 570px
   :height: 320px
   :scale: 100
   :alt: ROC of the model

   ROC of the model



.. figure:: /images/figure_2.png
   :align: center
   :width: 570px
   :height: 320px
   :scale: 100
   :alt:  Rrecision-Recall curve of the model

   Rrecision-Recall curve of the model



The result looks good. There is a question that we would like to answer: 


>>> Is the obtained binary portable or not?


The answer is "NOT". If you run it somewhere else, you will likely get something like this 

>>>   ./embedded: error while loading shared libraries: libpython2.7.so.1.0: cannot open shared object file: No such file or directory

or 

>>>  ./embedded:   'import site' failed; use -v for traceback


Can we fix the problem? 

Yes. We can.

Later in the tutorial,  I will demonstrate a tool, built on top of ``Go``, ``GCCGO`` and ``SWIG`` compilers which allows to build and package 
such Machine-Learning python script in one  portable  binary.
But before we test this tool, let's consider a few basic examples showing how one can obtain statically linked binaries from Python/C++ applications 
with help of the  ``GO`` compiler.


----------------------------------------
CGO: GO Binding for C/C++ 
----------------------------------------

As I said previously, ``GO`` makes  the `cross-compiling of code is very easy`__.  If you have ``GO>=1.5`` installed on Linux, there is no problem to build 
the windows ``*.exe`` binary from the GO code at this linux machine


.. code-block:: bash

   cat hello.go



>>>   package main
>>>
>>>   import "fmt"
>>>
>>>   func main() {
>>>        fmt.Printf("Hello\n")
>>>   }


.. code-block:: bash


     GOOS=windows GOARCH=386 go build -o hello.exe hello.go



That's it. Why not to try to exploit this nice feature of the ``GO`` to build cross-paltform applications from C++.


.. __ : https://github.com/golang/go/wiki/WindowsCrossCompiling  



XGO - CGO cross compiler
--------------------------------




For example, let's imagine, that I have developed a C++ program/API at the Linux machine. I want to start it at a computer running Microsoft Windows OS.
How to cross compile C++ for the MS Windows Machine without developing own C++ build toolchain of the compilers and libraries? Can we do it as easy as possible?

There is a tool called ``XGO``, `GO/CGO cross compiler`__, developed by Peter Szilagyi. Using LXC (Linux Containers) aka Docker containers, the ``XGO``
allows to build from ANY ``GO`` project  an executable for any platform supported by  ``GO``.  Here I highlight ANY because I want to stress that ANY includes 
also ``CGO`` projects.  In general, the ``CGO`` is a layer of stubs allowing to call C functions in GO environment. Also CGO allows to embed C++  code into the ``GO``
program. 

What doest it mean? 
We can release our C++ project for any platform and OS without big efforts, if we utilize ``XGO`` for the cross-compiling. 



.. __ : https://github.com/karalabe/xgo


But, first, let's try to play with the ``CGO``.  I have prepared a small sub-project ``cgo-cpp`` which shows basic ideas of binding C++ with the  ``GO`` runtime.
If you build the project, using the proposed ``Makefile``,


.. code-block:: bash

   cd cgo-cpp
   make



you will get a statically linked binary ``prog/prog-static`` which is executable on many linux-based 32/64-bit architecures. 

.. code-block:: bash

   ld prog/prog-static
   prog/prog-static


>>> 	not a dynamic executable
>>>     Generator is built
>>>     Characters: a A 
>>>     Decimals: 1977 650000
>>>     Preceding with blanks:       1977 
>>>     Preceding with zeros: 0000001977 
>>>     Some different radices: 100 64 144 0x64 0144 
>>>     floats: 3.14 +3e+00 3.141600E+00 
>>>     Width trick:    10 
>>>     A string 
>>>     1804289383

It looks promising. What is about ``XGO``? How can we use it  in this particular example? I have prepared another `sub-project`__, ``xgo-cpp`` being made as a replica of 
the ``cgo-cpp`` adopted  to ``XGO``. 


.. __ : https://bitbucket.org/iggy_floyd/xgo-cpp/


It is worth to mention that, to use the ``XGO``, and as a result, ``Docker`` containers,  we need to switch from the 32-bit VM to 64-bit VM.
All further tests and exercises are made in the VM with the following OS installed:


.. code-block:: bash


   uname -m
   lsb_release -a


>>>  x86_64
>>>  Distributor ID:	Debian
>>>  Description:	Debian GNU/Linux 8.1 (jessie)
>>>  Release:	8.1
>>>  Codename:	jessie



I suggest that you have installed the ``Docker`` system at your machine. If it's so, I recommend you to look at a few  ``"Getting started with XGO"`` instructions.


.. code-block:: bash

   cd xgo-cpp
   cat HOW-TO


The content of the ``HOW-TO`` is the following.



.. code-block:: bash 

   ### A simple HOW-TO on making cross-platform compilation with GO
   ### A docker system is required!
   ### 
   ###
   ###
   
   ## Installation of the GO tools: it uses user-defined paths of installation!
   ## To delete them, simply remove the folder ('xgo',here)!
   
   # build the temporary folders where we put our GO/XGO installation stuff
   mkdir -p xgo/go
   mkdir -p xgo/go-modules
   
   # download the latest go runtime environment
   wget https://storage.googleapis.com/golang/go1.6.2.linux-amd64.tar.gz -O xgo/go1.6.2.linux-amd64.tar.gz
   
   # unpack the go runtime environment
   tar -C xgo/ -xzf xgo/go1.6.2.linux-amd64.tar.gz
   
   
   # a simple script for initialization of the go runtime
   cat > xgo/ini_go.sh <<_END
   #!/bin/sh 
   
   export GOROOT=`pwd`/xgo/go
   export GOPATH=`pwd`/xgo/go-modules
   
   export PATH=\$PATH:\$GOROOT/bin:\$GOPATH/bin
   
   _END
   chmod a+rwx xgo/ini_go.sh
   
   # initialization of the GO and check version of the GO
   source xgo/ini_go.sh
   go version
   
   # get a XGO project  and check it
   go get github.com/karalabe/xgo
   ls $GOPATH/src/github.com/karalabe/xgo
   ls $GOPATH/bin
   
   
   # a smoke test of the XGO: build xgo/tests/embedded_cpp subproject of the XGO
   mkdir -p xgo/test
   cd xgo/test
   # build a statically linked program for 'linux/amd64' target
   sudo `which xgo` -go 1.6.1 -targets=linux/amd64  -ldflags '-extldflags "-static"'  -out embedded_cpp.0.0.1 github.com/karalabe/xgo/tests/embedded_cpp
 
   ...


As you see, it is quite easy to install the ``GO`` and ``XGO`` runtime. You can immediately test the ``XGO`` cross-compilation for different platforms like
``linux/amd64, linux/386, windows-6.0/*, ios-8.1/*`` etc.  The ``github.com/karalabe/xgo/tests/embedded_cpp`` project illustrates  embedded C++ code in ``GO`` sources. 
Let's try to compile it.



.. code-block:: bash

   cd xgo/test
   # testing the cross-platform compilation of  a smoke test from XGO
   targets="linux/amd64 linux/386 windows-6.0/* ios-8.1/*"
   for target in $(echo $targets | tr ' ' "\n" ); do echo $target;\ 
   sudo `which xgo` -go 1.6.1 -targets=$target -ldflags '-extldflags "-static"' -out \
   embedded_cpp.0.0.1 github.com/karalabe/xgo/tests/embedded_cpp; \
   done
   ls -rthl


These commands issue the output like it is shown below.



.. code-block:: bash

   ...

   Compiling for ios-8.1/arm-7...
   # github.com/karalabe/xgo/tests/embedded_cpp
   ./snippet.cpp:6:10: fatal error: 'iostream' file not found
    2016/04/29 15:29:44 Failed to cross compile package: exit status 2.

   -rwxr-xr-x 1 vagrant vagrant 3.5M Apr 29 15:29 embedded_cpp.0.0.1-linux-amd64
   -rwxr-xr-x 1 vagrant vagrant 3.1M Apr 29 15:29 embedded_cpp.0.0.1-linux-386
   -rwxr-xr-x 1 vagrant vagrant 2.6M Apr 29 15:29 embedded_cpp.0.0.1-windows-6.0-amd64.exe
   -rwxr-xr-x 1 vagrant vagrant 2.0M Apr 29 15:29 embedded_cpp.0.0.1-windows-6.0-386.exe


Everything runs smoothly except compilation for the ``ios-8.1/`` platform: it brings errors of missing C++ STL library in the ``XGO`` compiler toolchain for the ``OSX``.
Here are examples of the cross-platform building my simple ``xgo-cpp`` project. 

.. code-block:: bash

   cd xgo/test

   # linux: 64-bit architecture
   #

   # remove the cache of the previous exercise 
   sudo rm -r /root/.xgo-cache
   targets="linux/amd64"
   for target in $(echo $targets | tr ' ' "\n" ); do echo $target;\ 
   sudo `which xgo` -go 1.6.1 -targets=$target -ldflags '-extldflags "-static"' -out \
   xgo_example --deps=https://bitbucket.org/iggy_floyd/xgo-cpp/downloads/xgo-cpp.tar.bz2 bitbucket.org/iggy_floyd/xgo-cpp/;\
   done

   # linux: 32-bit architecture
   #
   targets="linux/386"
   for target in $(echo $targets | tr ' ' "\n" ); do echo $target;\
   sudo `which xgo` -v -x -go 1.6.1 -targets=$target -ldflags '-extldflags "-static"' -out \
   xgo_example --deps=https://bitbucket.org/iggy_floyd/xgo-cpp/downloads/xgo-cpp.tar.bz2 \
   --depsargs=-m32 bitbucket.org/iggy_floyd/xgo-cpp/;done
   
   ls -rthl
  ./xgo_example-linux-386 



And  this is the result.


.. code-block:: bash

   -rwxr-xr-x 1 vagrant vagrant 4.0M Apr 29 15:58 xgo_example-linux-386
   -rwxr-xr-x 1 vagrant vagrant 4.6M Apr 29 15:59 xgo_example-linux-amd64

   Generator is built
   Characters: a A 
   Decimals: 1977 650000
   Preceding with blanks:       1977 
   Preceding with zeros: 0000001977 
   Some different radices: 100 64 144 0x64 0144 
   floats: 3.14 +3e+00 3.141600E+00 
   Width trick:    10 
   A string 
   1804289383


It seems that ``XGO`` works :-).


----------------------------------------
Connecting GO and Python together
----------------------------------------

Now it is time to make a bridge between ``python`` and ``GO`` that we could use to 
compile python scripts in the cross-platform manner. OK. Let's go. 



Binding CGO and Cython together
----------------------------------------


First, we will try to use the same approach we have followed previously: we want to exploit ``CGO`` layer 
to connect ``*py`` scripts translated by ``Cython`` with the ``GO`` runtime in order to get a statically linked binary.


>>>  Python (script.py) --> Cython (script.c) --> CGO-->GO --> a statically linked binary ``script``


One can just have a look at another sub-project ``cgo-python``.
The stucture and Makefile of the ``cgo-python``look similar to ones in the ``cgo-cpp``.


.. code-block:: bash

   cd cgo-python
   make

   ldd prog/prog
   LD_LIBRARY_PATH=`pwd`/lib  prog/prog

As a result of the run, we got the output 

.. code-block:: bash


	linux-gate.so.1 =>  (0xb77b0000)
	libcgo-python.so => not found
	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0xb74c6000)
	libpthread.so.0 => /lib/i386-linux-gnu/i686/cmov/libpthread.so.0 (0xb74ad000)
	libdl.so.2 => /lib/i386-linux-gnu/i686/cmov/libdl.so.2 (0xb74a9000)
	libutil.so.1 => /lib/i386-linux-gnu/i686/cmov/libutil.so.1 (0xb74a5000)
	libm.so.6 => /lib/i386-linux-gnu/i686/cmov/libm.so.6 (0xb747f000)
	libstdc++.so.6 => /usr/lib/i386-linux-gnu/libstdc++.so.6 (0xb7392000)
	libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb722e000)
	libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb7215000)
	libgcc_s.so.1 => /lib/i386-linux-gnu/libgcc_s.so.1 (0xb71f8000)
	/lib/ld-linux.so.2 (0xb77b1000)

        Characters: a A 
        Decimals: 1977 650000
        Decimals: 1 
        Decimals: prog/prog 
        Preceding with blanks:       1977 
        Preceding with zeros: 0000001977 
        Some different radices: 100 64 144 0x64 0144 
        floats: 3.14 +3e+00 3.141600E+00 
        Width trick:    10 
        A string 

        fatal error: unexpected signal during runtime execution
        [signal 0xb code=0x2 addr=0x2 pc=0x8052cc4]
        runtime stack:
        runtime.throw(0x814a2c5)
 	/usr/local/go/src/pkg/runtime/panic.c:520 +0x71
        runtime.sigpanic()
 	/usr/local/go/src/pkg/runtime/os_linux.c:222 +0x46
        MHeap_Grow(0x81598e0, 0x10)
   
        ...


which explicitly shows that the **application binary interface (ABI)** produced by 
``Cython`` is not compatible with the ``GCO``. This indicates that the ``XGO``
cross-platform compiler can not be used to build ``cythonized`` files at the moment. 
In addition, there is another problem.
If you look at the `build toolchain`__ provided by the ``XGO``,  you realize that it mostly consists of 
the C/C++ libraries and C/C++ tools which are required for cross-platform 
compilation and linking of C/C++ programs.  This small extract from the
corresponding ``Dockerfile``of the ``XGO`` confirms my words.


.. code-block:: bash

   ...

   # Make sure apt-get is up to date and dependent packages are installed
   RUN \
   apt-get update && \
   apt-get install -y automake autogen build-essential ca-certificates            \
   gcc-5-arm-linux-gnueabi g++-5-arm-linux-gnueabi libc6-dev-armel-cross        \
   gcc-5-arm-linux-gnueabihf g++-5-arm-linux-gnueabihf libc6-dev-armhf-cross    \
   gcc-5-aarch64-linux-gnu g++-5-aarch64-linux-gnu libc6-dev-arm64-cross        \
   gcc-5-multilib g++-5-multilib gcc-mingw-w64 g++-mingw-w64 clang-3.7 llvm-dev \
   libtool libxml2-dev uuid-dev libssl-dev swig openjdk-7-jdk pkg-config patch  \
   make xz-utils cpio wget zip unzip p7zip git mercurial bzr texinfo help2man --no-install-recommends

   # Fix any stock package issues
   RUN \
   ln -s /usr/include/asm-generic /usr/include/asm && \
   ln -s /usr/bin/clang-3.7 /usr/bin/clang         && \
   ln -s /usr/bin/clang++-3.7 /usr/bin/clang++

   ...


There is no evidence of the presence of the python runtime which the ``cythonization`` depends on.  Perhaps I could fix this issue by adding python libraries to the 
build toolchain of the ``XGO``. I would rather postpone the deciding this question until my next tutorial on the topic. 



.. __ : https://raw.githubusercontent.com/karalabe/xgo/master/docker/base/Dockerfile


Is there any possibility to have a portable binary produced by the ``Cython``?

Yes. It is possible.

The next section is about the tool utilizing  the ``Cython``, ``SWIG``, ``GCCGO`` and ``my own packager``. They  all together build
the portable cross-platform binary from the  python script. 




----------------------------
Python Virtual Machine
----------------------------


I have called this tool  "Python Virtual Machine". Why is it "Virtual Machine"?
As you will see later, the result of the build chain is a package containing 
the executable and its context which includes all needed dependencies like dynamic libraries and 
python runtime environment.

The sketch of the toolchain is presented below.

>>>  Python (script.py) --> Cython (script.c) --> SWIG/GCCGO/GO --> my own packager --> a portable binary with the context

Let's clarify the unknown terms: ``SWIG`` and ``GCCGO``.

`Simplified Wrapper and Interface Generator (SWIG)`__ 
is a  tool that binds programs written in C and C++ with a variety of high-level programming languages.
Here we use the ``SWIG`` to write out the ``GO``-specific interface of the ``cythonized`` file ``(script.c)``.

Now it is time to tell about  the ``GCCGO`` compiler.
The ``GO`` language has always been defined by a spec, not an implementation. Hence
the ``GO`` team has adopted two different compilers that implement that spec: ``gcc`` and ``GCCGO``.

* ``gcc`` is the original compiler, and the ``GO`` uses it by default. 
* ``GCCGO`` is a different implementation with a different focus

Compared to the ``gcc``, the ``GCCGO``  slower to compile but supports more powerful optimization, 
so a CPU-bound program built by ``GCCGO`` will usually run faster.


The idea to use ``GCCGO`` instead of the ``CGO`` came to me after I read Yi Wang's  post `Statically Link C++ Code With Go Code`__.
I have slightly developed proposals of Yi Wang and added the ``Cython`` generator for getting a ``C/C++`` code from  a ``Python`` script.
Then the obtained ``C/C++`` file is processed by the ``SWIG/GCCGO/GO`` tools in the way as Yi Wang suggests.

.. __ : http://www.swig.org/
 
.. __ : https://cxwangyi.wordpress.com/2011/03/28/statically-linking-c-code-with-go-code/


.. Important::

   To run next exercise, One must install the ``SWIG`` generator. At the Ubuntu/Debian systems, it is usually done via ``sudo apt-get install swig``.


I have prepared the sub-project ``gccgo-python`` to demonstrate the technology on practice. One can simply change to the directory ```gccgo-python``
and build a binary through the ``make`` 


.. code-block:: bash

   cd gccgo-python
   make

As a result, we obtain the  dinamically linked binary ``gccgo-python`` which wrapps around ``src/simple_script.py``.
The script makes simple computation with the help of ``argparse``


.. obtained with ``./rsTCodeBlockProducer.sh python gccgo-python/src/simple_script.py``


.. code-block:: python 

   '''
   https://docs.python.org/2.7/library/argparse.html
   '''
   
   import argparse
   
   parser = argparse.ArgumentParser(description='Process some integers.')
   parser.add_argument('integers', metavar='N', type=int, nargs='+',
                      help='an integer for the accumulator')
   parser.add_argument('--sum', dest='accumulate', action='store_const',
                      const=sum, default=max,
                      help='sum the integers (default: find the max)')
   
   
   if __name__ == "__main__":
    args = parser.parse_args()
    print args.accumulate(args.integers)



If you list all dynamic libraries which executable loads, you will the presence of the ``libpython2.7.so.1.0``.

.. code-block:: bash 

   ldd ./gccgo-python
   ls -lh  ./gccgo-python


.. code-block:: bash

   	linux-gate.so.1 =>  (0xb7765000)
	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0xb747c000)
	libdl.so.2 => /lib/i386-linux-gnu/i686/cmov/libdl.so.2 (0xb7478000)
	libutil.so.1 => /lib/i386-linux-gnu/i686/cmov/libutil.so.1 (0xb7473000)
	libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb745a000)
	libgo.so.0 => /usr/lib/i386-linux-gnu/libgo.so.0 (0xb68d1000)
	libpthread.so.0 => /lib/i386-linux-gnu/i686/cmov/libpthread.so.0 (0xb68b8000)
	libm.so.6 => /lib/i386-linux-gnu/i686/cmov/libm.so.6 (0xb6892000)
	libgcc_s.so.1 => /lib/i386-linux-gnu/libgcc_s.so.1 (0xb6874000)
	libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb6710000)
	/lib/ld-linux.so.2 (0xb7766000)

        -rwxr-xr-x 1 debian debian 49K May  2 12:03 ./gccgo-python

This small binary can not be started at another machine. If you try to launch it somewhere else, you will probably get

>>> ./gccgo-python: error while loading shared libraries: libpython2.7.so.1.0: cannot open shared object file: No such file or directory


Now If we are repeating the same exercise with ``LDFLAGS=-static``, the statically linked binary will be issued


.. code-block:: bash

   LDFLAGS=-static make
   ldd ./gccgo-python
   ls -lh  ./gccgo-python



.. code-block:: bash

   ...

   not a dynamic executable

   -rwxr-xr-x 1 debian debian 4.5M May  2 12:09 ./gccgo-python


There are pros and cons to this. On the positive side, we can run the executable across different platforms. The negative effect is
increase of file's size. However, when you try to execute this file at the machine without ``python 2.7`` installed, you will come across the following
problem:


.. code-block:: bash

   ImportError: No module named site


This means that the python subroutines can't find modules needed to be imported. Can we fix it? Yes, we can. I have developed two hackish packagers
``build_package.sh`` and ``build_package_simplified.sh``. The latter is just a 'cutted' version  of the former. The shell scripts attempt to create an 
archive with the python runtime environment and executable packed together.


.. code-block:: bash

   cat build_package.sh


.. code-block:: bash 

   #!/bin/bash
   
   mkdir -p package_$1/.libs
   echo "dependences of the script... "
   cp  $1 package_$1/.libs
   ./statifier.py $1  package_$1/.libs
   
   # common modules
   echo "copying common modules..."
   mkdir -p package_$1/.libs/lib/
   cp -r /usr/lib/python2.7/  package_$1/.libs/lib/
   
   # scientific python modules and other stuff
   echo "copying scientific python modules and other stuff..."
   mkdir -p package_$1/.libs/local/lib/
   cp -r /usr/local/lib/python2.7/  package_$1/.libs/local/lib/   
   
   # numpy hack!
   # add numerical support for numpy!
   echo "adding numerical support for numpy!..."
   cp /usr/lib/libblas.so.3 package_$1/.libs/
   ./statifier.py /usr/lib/libblas.so.3 package_$1/.libs/
   cp /usr/lib/liblapack.so.3  package_$1/.libs/
      
   # a hack to add hashlib support
   echo "hacking to add hashlib support..."
   ls /usr/lib/python2.7/lib-dynload/  | xargs -I {} ./statifier.py /usr/lib/python2.7/lib-dynload/{} package_$1/.libs/
   
   # a hack to add matlab support
   echo "hacking to add matplotlib support..."
   ls /usr/lib/python2.7/dist-packages/gi/  | grep .so | xargs -I {} ./statifier.py /usr/lib/python2.7/dist-packages/gi/{} package_$1/.libs/
   #cp -r /usr/share/pyshared/gi/   /usr/lib/python2.7/dist-packages/
   rsync /usr/share/pyshared/gi package_$1/.libs/lib/python2.7/dist-packages/  -a --copy-links -v
   rsync /usr/share/pyshared/wx-2.8-gtk2-unicode/wx package_$1/.libs/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode/  -a --copy-links -v
   rsync /usr/share/pyshared/wx-2.8-gtk2-unicode/wxPython package_$1/.libs/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode/  -a --copy-links -v
   
   echo "hacking Tornado Service of the WebAgg..."
   sed -i "706s/\(.*\)/#\1/"   package_$1/.libs/local/lib/python2.7/dist-packages/tornado/ioloop.py
   sed -i "707s/\(.*\)/#\1/"   package_$1/.libs/local/lib/python2.7/dist-packages/tornado/ioloop.py
   
   
   
   echo "writting the executable script..."
   cat  > package_$1/$1 <<_EOF
   #!/bin/bash
   
   _abs_path_to_this_file=\$(readlink -f "\$0")
   _local_dir=\$(dirname "\$_abs_path_to_this_file")
   _libs_dir="\${_local_dir}/.libs"
   
   export LD_LIBRARY_PATH=\${_libs_dir}:\$LD_LIBRARY_PATH
   export PYTHONPATH=\${_libs_dir}/lib/python2.7/dist-packages/:\${_libs_dir}/lib/python2.7/:\${_libs_dir}/local/lib/python2.7/dist-packages/:\${_libs_dir}/local/lib/python2.7/:\$PYTHONPATH
   export PYTHONHOME=\${_libs_dir}
      
   .libs/$1 \$@
   
   _EOF
   chmod a+x package_$1/$1
         
   # copy train.csv
   cp train.csv package_$1
   echo "archiving..."
   tar -zcf package_$1.tgz package_$1 



Here is one more remark: to have a really portable binary, the python script should follow a few rules:

* a workaround, to escape native python modules on the target machine, should be implemented before imports of system-specific modules like ``sklearn``, ``pandas``, ``matplotlib`` etc;

* the ``matplotlib`` should use ``WebAgg`` engine to render graphics. The standard matplotblib's  backend ``GTK`` is not portable across OS.

The version of the well known script ``titanic_logistic_regression_grid_search_roc_plot_example.py``, where the above points are taken into account, can be found
in the directory ``gccgo-python/src``.


.. code-block:: python 

   
   #!/usr/bin/env python
   
   ''' This is an example of the application of the Logistic fucntion to the dataset analysis. 
       Logistic function is helpful to model binary models
   
       This is a analysis with dedicated sklearn sub-modules.
   
       The basic idea is taken from 
       http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#example-linear-model-plot-iris-logistic-py
       http://scikit-learn.org/stable/auto_examples/grid_search_digits.html	
       http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/
   
       DataFrameImputer was taken from 
       http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
   
       About Precision and Recall
       http://en.wikipedia.org/wiki/Precision_and_recall
   
       About Type 1 and Type 2 errors 
       http://en.wikipedia.org/wiki/Type_I_and_type_II_errors
   
       About scorring parameters for sklearn classifiers
       http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
   
       About plottig ROC in sklearn
       http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html
   
       About plottig Precision vs Recall in sklearn
       http://scikit-learn.org/stable/auto_examples/plot_precision_recall.html
   
   '''
   import os
   import sys
   import pprint
   sys.path = ['/usr/local/lib/python2.7/dist-packages'] + sys.path # to fix the problem with numpy: this replaces  1.6 version by 1.9
   
   # workaround to escape native python modules on the target machine
   try:
     pythonpath = os.environ['PYTHONPATH']
     for i,item in enumerate(pythonpath.split(":")):
      sys.path.insert(i,item)
   except  KeyError,e:
     pprint.pprint(e)
   
   pprint.pprint(sys.path)
   
   import pandas as pd
   from sklearn.ensemble import ExtraTreesClassifier
   from sklearn.cross_validation import cross_val_score
   from sklearn.preprocessing import Imputer
   from sklearn import preprocessing
   import numpy as np
   from sklearn.base import TransformerMixin
   from StringIO import StringIO
   import prettytable    
   import statsmodels.api as sm
   #  Gtk and Wx are not cross-platformed. Use WebAgg for matplotlib backend!
   import matplotlib
   matplotlib.use('WebAgg')

   ...


To make the portable binary package of the Machine-Learning script, you execute several commands as shown below.


.. code-block:: bash

   cd gccgo-python/
   make clean
   PYSRC=src/titanic_logistic_regression_grid_search_roc_plot_example.py make
   make build-package
   ls -lh package_gccgo-python.tgz
  

You will see the output of the previous command if everything runs smoothly.

.. code-block:: bash

   ...
   hacking Tornado Service of the WebAgg...
   writting the executable script...
   archiving...

  -rw-r--r-- 1 debian debian 358M May  2 13:00 package_gccgo-python.tgz




-------------------
Static-Python
-------------------

This is `a fork`__ of the official Python hg repository with additional tools to enable building Python for static linking.

.. __ : https://github.com/bendmorris/static-python


Using the ``Static-Python``, we can

* run Python programs on other machines without requiring that they install Python.
* run Python programs on other machines without requiring that they have the same versions of the same libraries installed that you do.



Unfortunately I have to remark that ``Static-Python`` doesn't support ``python scientific`` extensions and modules like ``numpy``, ``pandas`` etc.
Only 'simple' modules (pure pythonic) could be invoked in the statically linked python environment. The simple ``HOW-TO`` on usage of the package is presented
below.

.. code-block:: bash

   git clone https://github.com/bendmorris/static-python
   cd static-python
   git checkout 2.7
   sudo apt-get install  python-gdbm
   sudo apt-get install python-tk
   # change add-ons to the BUILTINS variable in Static.make They should be defined as give below
   # override BUILTINS+= array cmath math _struct time  _random operator _collections _heapq itertools _functools _elementtree   _bisect unicodedata atexit _weakref datetime
   nano Static.make
   make -f Static.make BUILTINS="math"
   ldd ./python


>>>  not a dynamic executable

We can test the ``math`` module, if execute the following command.


.. code-block:: bash

   ./python  -c "import math; print math.pi"


>>>  3.14159265359


------------------------------------------------------------------------------------------------------------------
Bonus: A simple statifier to make a portable executable package from any binary
------------------------------------------------------------------------------------------------------------------


There are several statifiers on the marker:

* `ELF statifier`__ 
* `Ermine`__

None of them suits well if need a free of charge, out-of-box solution.

``ELF statifiers`` requires to switch off the address randomization in the kernel,


.. code-block:: bash

   cat /proc/sys/kernel/randomize_va_space
   echo -n 0 > /proc/sys/kernel/randomize_va_space    
   statify xxxxx yyyyyy
   echo -n 2 > /proc/sys/kernel/randomize_va_space

while ``Ermine`` costs money. 

.. __ : https://sourceforge.net/projects/statifier/

.. __ : http://magicermine.com/


I have written a simple shell script to pack any executable together with the runtime environment. 
It is called ``binary_statifier``. You might probe it via the executing a set of commands as shown below.

.. code-block:: bash

   # let's build a portable package for the  LateX system
   cp `which latex` .
   ./build_package.sh latex
   rm latex


Then we can extract the package at other machine and run it.

.. code-block:: bash

   # another platform
   tar -xf package_latex.tgz
   cd package_latex
   ./latex


The output of the last command might be like below.


.. code-block:: bash

   warning: kpathsea: configuration file texmf.cnf not found in these directories: /etc/texmf/web2c:/usr/share/texlive/texmf/web2c:/usr/share/texlive/texmf-dist/web2c:/usr/local/share/texmf/web2c.
   This is pdfTeX, Version 3.1415926-2.4-1.40.13 (TeX Live 2012/Debian)
   **
   ! End of file on the terminal... why?








----------------------------------------------------------------
Documentation
----------------------------------------------------------------

The main parts of documentation are placed  in the ``README.rst``.
Running the command 

.. code-block:: bash
    
    make doc
    
    
will make a folder ``doc/`` with the ``index.html`` which presents documention  in
the HTML format. One can start httpd service to see the documention at the ``localhost:8010``.


.. code-block:: bash

    make serve








----------------------------------------------------------------
References
----------------------------------------------------------------


.. target-notes::
    


