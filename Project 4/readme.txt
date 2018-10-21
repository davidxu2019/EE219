EE219 Project 4

Winter 2018

Teammembers:
Fangyao Liu----UCLA ECE Graduate student
	       UID:204945018
	       Email:fangyao@okstate.edu
Xuan Hu    ----UCLA ECE Graduate student
	       UID:505031796
	       Email:x1310317@gmail.com
YanZhe Xu  ----UCLA ECE Graduate student
	       UID:404946757
	       Email:xuyanzhesjtu@g.ucla.edu
Zhechen Xu ----UCLA ECE Graduate student
	       UID:805030074
	       Email:xzhechen@ucla.edu

Introduction:  We use a Network backup Dataset, which is comprised of simulated trac data on a backup system over a network. The system monitors the files residing in a destination machine and copies their changes in four hour cycles. At the end of each backup process, the size of the data moved to the destination as well as the duration it took are logged,to be used for developing prediction models. We done a workflow as a task that backs up data from a group of files, which have similar patterns of change in terms of size over time. The dataset has around 18000 data points with the following columns/variables:
Week index
Day of the week at which the file back up has started
Backup start time: Hour of the day
Workow ID
File name
Backup size: the size of the file that is backed up in that cycle in GB
Backup time: the duration of the backup procedure in hour 



IMPLEMENTATION
    Environment:
        python 2.7
    Dependencies:
	a.sklearn v0.19.1
	b.matplotlib v2.1.0
	c.numpy v1.13.3

Usage:
put util.py,question_a.py,questionb.py,question_c_d.py,question_e.py and network_backup_dataset.csv together.
$python quesion_a.py
$python quesion_b.py
$python quesion_c_d.py
$python quesion_e.py