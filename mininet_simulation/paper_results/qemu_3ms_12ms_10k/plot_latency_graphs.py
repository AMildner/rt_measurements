#!/usr/bin/python

"""
    Plot the results from the latency tests for the RT-based Protocol

    @Author: Alexander Mildner
    @Email: mildner@in.tum.de
    2018

"""
from model_rt import RTModel
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
    Example input
    Files with name according to the Input Params:
    <num_max_rts>_<num_max_pkts>/<loss_prob>_<num_max_rts>_<num_max_pkts>_<link_delay>

    Contain the following set of data:

    SEQ,NUM_RTS,LATENCY,START_TIME,END_TIME,EST_RTT

"""


if __name__ == "__main__":

    input_folder = sys.argv[1]
    num_max_rts, num_max_pkts, delay, timeout = input_folder.split("_")

    loss_probs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    example = [0.2, 0.5, 1.5, 0.2, 1.1, 0.1, 3.0, 1.1, 2.8, 0.5]

    max_latency = []
    avg_latency = []
    print sys.argv[2]

    if int(sys.argv[2]) is 0:
        for loss in loss_probs:
            #0_1_1000_30_120000.data
            with open("{}/{}_{}_{}_{}_{}.csv".format(input_folder, loss, num_max_rts, num_max_pkts, delay, timeout), "r") as file :
                latencies = []
                for line in file:
                    seq, rts, tot_latency, start_time, end_time, received = line.split(',')
                    #if float(tot_latency) == 0:
                    #    tot_latency = float(end_time) - float(start_time)
                    #if float(tot_latency) > 0.8:
                    #    print "FOUND OUTLIER AT SEQ: {}".format(seq)
                    #elif float(tot_latency) > 0 and float(tot_latency) < 0.5:
                    latencies.append(float(tot_latency)*1000.0)
                max_latency.append(max(latencies))
                avg_latency.append(sum(latencies)/float(len(latencies)))

        print max_latency
        plt.plot(loss_probs, max_latency, "bo-", label="max latency")
        plt.plot(loss_probs, avg_latency, "ro-", label="avg latency")
        plt.axis([0, 90, 0, 800])
        plt.xlabel("Loss Probability [%]")
        plt.ylabel("maximum latency [msec]")
        plt.grid(True)
        plt.legend()
        plt.title("RTs={}, delay={}ms, timeout={}ms".format(num_max_rts, delay, int(timeout)/1000))
        plt.savefig('./plots/max_latency_c_{}_{}.pdf'.format(num_max_rts, delay))

    elif int(sys.argv[2]) is 1:
        list_max_num_exceeded = []

        for loss in loss_probs:
            with open("{}/{}_{}_{}_{}.csv".format(input_folder, loss, num_max_rts, num_max_pkts, delay, timeout), "r") as file :
                num_exceeded_rts = 0
                for line in file:
                    seq, rts, tot_latency, start_time, end_time, est_rtt = line.split(',')
                    if int(rts) > int(num_max_rts):
                        num_exceeded_rts += 1
                list_max_num_exceeded.append(num_exceeded_rts)

        ind = np.arange(len(loss_probs))
        plt.bar(ind, height=list_max_num_exceeded)
        plt.xticks(ind, ('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'))
        plt.xlabel("Loss Probability")
        plt.ylabel("number max RTs exceeded")
        #plt.grid(True)
        plt.savefig('./plots/max_rts_exceeded_{}_{}.pdf'.format(num_max_rts, delay))

    elif int(sys.argv[2]) is 2:
        latencies_box = []

        for loss in loss_probs:
            with open("{}/{}_{}_{}_{}.csv".format(input_folder, loss, num_max_rts, num_max_pkts, delay, timeout), "r") as file :
                latencies = []
                for line in file:
                    seq, rts, tot_latency, start_time, end_time, est_rtt = line.split(',')
                    #if int(rts) < int(num_max_rts):
                    latencies.append(float(tot_latency)*1000.0)
                #latencies_box.append(latencies)

            fig = plt.figure(1, figsize=(18, 12))
            ax = fig.add_subplot(111)
            bp = ax.boxplot(latencies)
            ax.get_xaxis().tick_bottom()
            ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90'])
            plt.savefig('./plots/latency_box_{}_{}_{}.pdf'.format(num_max_rts, delay, loss))
            plt.close(fig)

    elif int(sys.argv[2]) is 3:
        latencies_box = []
        num_flows = []

        for loss in loss_probs:
            with open("{}/{}_{}_{}_{}.csv".format(input_folder, loss, num_max_rts, num_max_pkts, delay, timeout), "r") as file :
                latencies = []
                for line in file:
                    seq, rts, tot_latency, start_time, end_time, est_rtt = line.split(',')
                    latencies.append(float(tot_latency)*1000.0)
                latencies_box.append(latencies)
                num_flows.append(np.arange(0 ,len(latencies), 1))
        for i, it in enumerate(num_flows):
            fig = plt.figure(1, figsize=(18, 12))
            ax = fig.add_subplot(111)
            ax.plot(num_flows[i], latencies_box[i])
            plt.savefig('./plots/latency_graph_{}_{}_{}.pdf'.format(num_max_rts, delay,i))
            plt.close(fig)

    elif int(sys.argv[2]) is 4:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        plots = [ax1, ax2, ax3, ax4]
        i = 0
        for loss in loss_probs:
            #0_1_1000_30_120000.data
            with open("{}/{}_{}_{}_{}_{}.csv".format(input_folder, loss, num_max_rts, num_max_pkts, delay, timeout), "r") as file :
                recv_lat = []
                not_recv_seq = []
                recv_seq = []
                not_recv_lat = []
                for line in file:
                    seq, rts, tot_latency, start_time, end_time, received = line.split(',')
                    if int(received) is 0:
                        not_recv_seq.append(int(seq))
                        not_recv_lat.append(float(tot_latency)*1000.0)
                    else:
                        recv_seq.append(int(seq))
                        recv_lat.append(float(tot_latency)*1000.0)
                real_loss = (float(len(not_recv_seq))/1000.0)*100.0
                plots[i].scatter(not_recv_seq, not_recv_lat, marker='x', color='r', s=7)
                plots[i].scatter(recv_seq, recv_lat, marker='o', color='b', s=7)
                plots[i].set_title("{} RTs, p={}%, real_loss={}%".format(num_max_rts, loss, real_loss))
                i += 1
        plots[2].set_xlabel("Seqence number")
        plots[3].set_xlabel("Seqence number")
        plots[0].set_ylabel("Latency [ms]")
        plots[2].set_ylabel("Latency [ms]")
        plt.show()

    if int(sys.argv[2]) is 5:
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        colors_lat = ['b^-', 'ro-', 'gx-', 'ks-']
        colors_avg = ['b^--', 'ro--', 'gx--', 'ks--']
        colors_delay = ['b^:', 'ro:', 'gx:', 'ks:']
        i = 0
        rts_list = [1, 2, 3, 4]

        # RT Model Params
        params = [0.1, 0.1, 1, 2, 3, 0.001, 12, 3]

        for rtss in rts_list:
            max_latency = []
            avg_latency = []
            tmp_delay = []
            params[7] = rtss
            for loss in loss_probs:
                #0_1_1000_30_120000.data
                with open("{}_{}_{}_{}/{}_{}_{}_{}_{}.csv".format(rtss, num_max_pkts, delay, timeout, loss, rtss, num_max_pkts, delay, timeout), "r") as file :
                    latencies = []
                    for line in file:
                        seq, rts, tot_latency, start_time, end_time, received = line.split(',')
                        #if (int(received) is not 0 and rtss is not 2):
                        #    latencies.append(float(tot_latency)*1000.0)
                        #else:
                        #    latencies.append(float(tot_latency)*1000.0)
                        latencies.append(float(tot_latency)*1000.0)
                    max_latency.append(max(latencies))
                    avg_latency.append(sum(latencies)/float(len(latencies)))
                tmp_model = RTModel(params)
                tmp_model.set_p_param(float(loss)/100.0)
                tmp_model.calculate()
                # Test, adding maximum feedback delays to the delay bounds
                if (tmp_model.check_stability()):
                    print "OK"
                else:
                    print "Not OK"
                #tmp_delay.append(tmp_model.delay_bound + float(rtss+1)*float(tmp_model.W))
                tmp_delay.append(tmp_model.delay_bound + float(rtss+1)*float(tmp_model.W))
            ax.plot(loss_probs, tmp_delay, colors_delay[i], label="NC delay Bound, RTs={}".format(rtss))
            ax.plot(loss_probs, max_latency, colors_lat[i], label="max latency, RTs={}".format(rtss))
            #ax.plot(loss_probs, avg_latency, colors_avg[i], label="avg latency RTs={}".format(rtss))
            #print tmp_delay
            i += 1
        plt.axis([0, 90, 0, 160])
        plt.xlabel("Loss Probability [%]")
        plt.ylabel("maximum latency [msec]")
        plt.grid(False)
        plt.legend(prop={'size': 9}, loc="upper left", ncol=2)
        plt.title("num_flows={}, delay={}ms, timeout={}ms".format(num_max_pkts, delay, int(timeout)/1000))
        plt.savefig('./plots/max_latency_rts_c_{}_{}_SNC.pdf'.format(num_max_rts, delay), bbox_inches="tight")
        #plt.show()

    elif int(sys.argv[2]) is 6:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        plots = [ax1, ax2, ax3, ax4]
        i = 0
        rts_list = [1, 2, 3, 4]
        loss = 20
        for rtss in rts_list:
            #0_1_1000_30_120000.data
            with open("{}_{}_{}_{}/{}_{}_{}_{}_{}.csv".format(rtss, num_max_pkts, delay, timeout, loss, rtss, num_max_pkts, delay, timeout), "r") as file :
                recv_lat = []
                not_recv_seq = []
                recv_seq = []
                not_recv_lat = []
                for line in file:
                    seq, rts, tot_latency, start_time, end_time, received = line.split(',')
                    if int(received) is 0:
                        not_recv_seq.append(int(seq))
                        not_recv_lat.append(float(tot_latency)*1000.0)
                    else:
                        recv_seq.append(int(seq))
                        recv_lat.append(float(tot_latency)*1000.0)
                real_loss = (float(len(not_recv_seq))/float(num_max_pkts))*100.0
                plots[i].scatter(not_recv_seq, not_recv_lat, marker='x', color='r', s=7)
                plots[i].scatter(recv_seq, recv_lat, marker='o', color='b', s=7)
                plots[i].set_title("{} RTs, p={}%, pkt_loss={}%".format(rtss, loss, real_loss), fontsize=10)
                i += 1
        plots[2].set_xlabel("Seqence number")
        plots[3].set_xlabel("Seqence number")
        plots[0].set_ylabel("Latency [ms]")
        plots[2].set_ylabel("Latency [ms]")
        plt.savefig("./plots/scatter_compare_{}_{}_{}_{}_1-4_final.pdf".format(delay, num_max_pkts, timeout, loss), bbox_inches="tight")

    if int(sys.argv[2]) is 7:
        colors_lat = ['b^-', 'ro-', 'gx-', 'ks-']
        rts_list = [1,2,3,4]
        p = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

        for rtss in rts_list:
            for loss in loss_probs:
                #0_1_1000_30_120000.data
                with open("{}_{}_{}_{}/{}_{}_{}_{}_{}.csv".format(rtss, num_max_pkts, delay, timeout, loss, rtss, num_max_pkts, delay, timeout), "r") as file :
                    latencies = []
                    for line in file:
                        seq, rts, tot_latency, start_time, end_time, received = line.split(',')
                        latencies.append(float(tot_latency)*1000.0)
                    np.asarray(latencies)
                    d = np.sort(latencies)
                    perc = mlab.prctile(d, p=p)
                    plt.plot(d, label="p={}".format(loss))
            plt.xticks((len(d)-1) * p/100., map(str, p))
            plt.title("num_flows={}, delay={}ms, timeout={}ms".format(num_max_pkts, delay, int(timeout)/1000))
            plt.legend()
            plt.show()
        #plt.savefig('./plots/max_latency_rts_c_{}_{}_SNC.pdf'.format(num_max_rts, delay), bbox_inches="tight")
        #plt.show()

    exit(0)
