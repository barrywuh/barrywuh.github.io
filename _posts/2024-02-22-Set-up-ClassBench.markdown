---
layout: post
title:  "Set up ClassBench"
date:   2024-02-22 12:22:47 +1300
categories: networking tools
---
[ClassBench] is a legacy program which generates filter sets and traces matching these filter sets. It was created almost two decades ago and not maintained any longer. In spite of this, many researchers in the academia still use it for their experiments related to router/switch. You can download its source code and try to run it. But since it was developed 20 years ago, you will probably run into a lot of errors (compilation ones and others). I was in the same situation and I asked around for help. Finally, a friend who is a Linux expert helped me. His suggestion is to set up an environment as if we were in early 2000's. We tried separately and it works. 

The rough idea is to set up a legacy Linux as a virtual machine and compile the souce code and run the program there. It seems to be quite easy to do but it turned out to be not easy. The main challenge comes from we need a way to transfer the source files into the legacy Linux whose web browser is too old and it cannot download the source files directly from a website. Eventually,we use SCP to bypass this issue. 

Here is a list of steps you can follow:
1.  Download an old Ubuntu server from [Ubuntu_old_release].
1.  Create a VM in VirtualBox with the downloaded .iso file 
1.  In order to transfer files from your host OS to the guest OS (legacy Linux), we need to set up SSH server on Linux.
{% highlight ruby %}
sudo apt-get install openssh-server
#You will be asked to insert a CD-rom. Click the CD-ROM icon and choose the iso file you used for installing the Linux server. 
{% endhighlight %}
1.  Configure the Virtual Box with port forwarding. In Virtual Box, choose Settings>>Network>>Advanced>>port forwarding; add a new rule "ssh" for ssh:
{% highlight ruby %}
    host IP: 0.0.0.0
    host port: 2233
    guest IP: blank
    guest port:22
#NAT is used for Adapter 1: attached to NAT
{% endhighlight %}
1.  SCP files from host (e.g., a windows laptop) to the `guest` Linux, assuming you've downloaded db_generator.tar.gz and parameter_files.tar.gz from [ClassBench].
{% highlight ruby %}
scp -P 2233 db_generator.tar.gz user_name@127.0.0.1:/home/user_name
#Replace user_name as your user name for the Linux
#scp all other files similarly;
{% endhighlight %}
1.  On (guest Linux) unzip all the files received
{% highlight ruby %}
tar -zxvf db_generator.tar.gz
tar -zxvf parameter_files.tar.gz
{% endhighlight %}
1.  Install make and other packages on guest OS
{% highlight ruby %}
sudo apt-get install make
sudo apt-get install g++
sudo apt-get install build-essential
{% endhighlight %}
1.  Modify makefile under db_generator (a folder)
{% highlight ruby %}
CFLAGS = -g -pg
##CFLAGS = -O2
{% endhighlight %}
1.  Use "make all" to compile db_generator
{% highlight ruby %}
make all
{% endhighlight %}
1.  Use `db_generator` to generate a file test1000acl
{% highlight ruby %}
db_generator -bc ../parameter_files/acl1_seed 10000 2 -0.5 0.1 test1000acl
{% endhighlight %}
1.  On `host` OS copy test1000acl from the Linux
{% highlight ruby %}
  scp -P 2233 user_name@127.0.0.1:/home/user_name/db_generator/test1000acl .
{% endhighlight %}

[Ubuntu_old_release]: https://old-releases.ubuntu.com/releases/dapper/ubuntu-6.06.2-server-i386.iso
[ClassBench]: https://www.arl.wustl.edu/classbench/
