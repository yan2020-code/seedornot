# Seedornot Fuzzer       


## Work Flow

1. SRC + Sanitizer (use wllvm) => prog + prog.bc
2. prog.bc (use afl-clang-fast) => afl-prog
3. prog.bc (use dyn-instr-clang) => dyninst-prog + dyninst-prog.bc
4. dyninst-prog.bc (use DMA) => prog.edge + prog.reach.bug + prog.reach.cov

Eventually, update config file 
AFL fuzz with *afl-prog*
- fill in [moriarty]
    - target_bin (remember to put in proper cli options as well!)
QSYM run with *prog*
- fill in [qsym conc_explorer]
    - cmd (remember to put in proper cli options as well!)
Coordinator use *dyninst-prog* + *prog.edge* + *prog.reach.bug* + *prog.reach.cov* to replay and collect data  
- fill in [auxiliary info]
    - replay_prog_cmd (remember to put in proper cli options as well!)
    - bbl_bug_map
    - bbl_cov_map
    - pair_edge_file


To see some examples on how to configure Seedornot, please look at the [examples](./examples) directory.  

## How to build Seedornot

### Build with Docker
```
$ curl -fsSL https://get.docker.com/ | sudo sh  

$ sudo usermod -aG docker [user_id]

$ docker run ubuntu:16.04

Unable to find image 'ubuntu:16.04' locally  
16.04: Pulling from library/ubuntu
Digest: sha256:e348fbbea0e0a0e73ab0370de151e7800684445c509d46195aef73e090a49bd6  
Status: Downloaded newer image for ubuntu:16.04  

$ docker build -t seedornot .  

$ docker images


```
# use --privileged flag to allow docker to mount a ramdisk

$ docker run --cap-add=SYS_PTRACE --priviledged -it seedornot:latest /bin/bash    
```
  

