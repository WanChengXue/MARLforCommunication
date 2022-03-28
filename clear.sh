ps -ef | grep Learner | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep plasma | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep tensorboard | grep -v grep | awk '{print "kill "$2}' | sh
ps -ef | grep Worker | grep -v grep | awk '{print "kill "$2}' | sh
rm -rf logs
# rm -rf Exp/Model/model_pool/*
rm -rf nohup.out
# rm -rf Exp/Result
rm -rf Exp/Model/saved_model/*
rm -rf Worker/Download_model/*

# ps -ef | grep http_server_process | grep -v grep | awk '{print "kill "$2}' | sh
