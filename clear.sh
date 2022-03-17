rm -rf logs
rm -rf Exp/Modee/model_pool/*
rm -rf Worker/Download_model/*
ps -ef | grep Learner | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep plasma | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep tensorboard | grep -v grep | awk '{print "kill "$2}' | sh