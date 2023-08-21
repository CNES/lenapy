Module lenapy.sandbox
=====================

Functions
---------

    
`to_difgri(data, dir_out, prefix, suffix)`
:   difgri format use in gins tool is a non binary format with a specific number of columns and format should be %+13.6e
    for example for 64800 values ( 1deg. x 1deg. ) there are 6480 lines sorted by row from left to right,
    starting from longitude -179.5, latitude 89.5.
    (The first 360 values concern latitude 89.5, the next 360 latitude 88.5, etc.)