#!/usr/bin/python

from mininet.net import Mininet
from mininet.log import *
from mininet.cli import CLI
from mininet.link import TCLink
import os
import time

if __name__ == '__main__':

    delays = [30]
    loss_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    max_pkts = [1000, 10000]
    max_rts = [1, 2, 3, 4, 5]
    timeout = 120000

    net = Mininet(link=TCLink)

    net.addHost('h1', ip="10.0.0.1", mac="00:00:00:00:00:01")
    net.addHost('h2', ip="10.0.0.2", mac="00:00:00:00:00:02")
    # net.addSwitch('s1')

    h1 = net['h1']
    h2 = net['h2']
    # s1 = net.switches[0]

    params = dict(intfName1='eth0', intfName2='eth0', delay='0ms', loss=0)

    net.addLink(h1, h2, **params)
    # net.addLink(h1, s1, loss=5)
    # net.addLink(h2, s1, loss=5)

    setLogLevel( 'info' )
    net.start()
    #net.startTerms()
    for delay in delays:
        for max_pkt in max_pkts:
            for max_rt in max_rts:
		current_dir = "c_results/{}_{}_{}_{}".format(max_rt, max_pkt, delay, timeout)
		h1.cmd("mkdir {}".format(current_dir))
                for loss in loss_list:
                    info("*** Working on Run: {} {} {}\n".format(loss, max_rt, max_pkt, delay))
                    info("*** Setting Loss and Delay\n")
                    h1.cmd("tc qdisc change dev eth0 root netem delay {}ms loss {}%".format(delay, loss))
                    h2.cmd("tc qdisc change dev eth0 root netem delay {}ms loss {}%".format(delay, loss))
                    time.sleep(0.2)
                    info("*** Starting Test ***")
                    h2.cmd("./rt_proto_udp -r -m{} -n{} -t{} &".format(max_rt, max_pkt, timeout))
		            time.sleep(0.1)
                    h1.sendCmd("./rt_proto_udp -s -m{} -n{} -t{} > {}/{}_{}_{}_{}_{}.data".format(max_rt, max_pkt, timeout, current_dir, loss, max_rt, max_pkt, delay, timeout))
		            result = h1.waitOutput()
                    if result.rstrip() is not None:
                        h2.cmd('pkill rt_proto_udp')
			            h2.sendCmd('ps aux | grep rt_proto_udp')
                        psaux = h2.waitOutput()
                        print psaux
		                info("Test finished : " + str(result))
		                info("*** Finished Run: {} {} {} {} {}\n".format(loss, max_rt, max_pkt, delay, timeout))
                    else:
                        info("Approach failed!\n")
                #os.system("python plot_latency_graphs.py {}_{}_results {} 0".format(max_rt, max_pkt, delay))
                #os.system("python plot_latency_graphs.py {}_{}_results {} 1".format(max_rt, max_pkt, delay))
    #CLI(net)
    net.stop()
