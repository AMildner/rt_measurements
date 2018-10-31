/*
 *  This is an implementation of the RT-Protocol using C/C++.
 *  It is intended for research purposes only, and thus comes without any warranty.
 *
 *
 *  @Author: Alexander Mildner
 *  @Email: mildner@in.tum.de
 *
 */

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <arpa/inet.h>
#include <iostream>
#include <string.h>
#include <cstdlib>
#include <netinet/in.h>
#include <time.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>

using namespace std;

#define RECEIVER_MODE    true
#define SENDER_MODE      false
#define SPORT            60001
#define DPORT            60002
#define RECIEVER_IP      "10.0.0.2"
#define SENDER_IP        "10.0.0.1"
#define MAX_BUF_SIZE     512 //65536
static int num_pkts;
static int max_rt;
static bool mode;
static long timeout_usec;

struct payload {
    timespec timestamp;
    uint64_t sequenceNum;
    uint8_t isAck;
};

timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}
// other approach
static long diff_in_ns(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000000.0 + diff.tv_nsec);
}


static void usage(char *progname)
{
    fprintf(stderr,
            "\n"
            "usage: %s [options]\n"
            "\n"
            "-n [num]        Number of packets to send\n"
            "-h              Prints this message and exits\n"
            "-s              Run in Sender Mode\n"
            "-r              Run in Receiver Mode\n"
            "-m [num]        Number of Max Retransmissions\n"
            "-t [num]        Timeout value [usec]\n "
            "\n",
            progname); //Default values here
}


int main(int argc, char *argv[]) {

    int c;
    char *progname;

    /* Process CLI arguments */
    progname = strrchr(argv[0], '/');
    progname = progname ? 1 + progname : argv[0];
    while (EOF != (c = getopt(argc, argv, "n:m:t:srh"))) {
        switch (c) {
            case 't':
                timeout_usec = atoi(optarg);
                break;
            case 'n':
                num_pkts = atoi(optarg);
                break;
            case 's':
                mode = SENDER_MODE;
                break;
            case 'r':
                mode = RECEIVER_MODE;
                break;
            case 'm':
                max_rt = atoi(optarg);
                break;
            case 'h':
                usage(progname);
                return 0;
            case '?':
                usage(progname);
                return -1;
        }
    }

    // Create Sockets

    int sock_s;
    sock_s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock_s < 0) {
        cout << "ERROR creating socket! Exiting.. RETURN: " << sock_s << endl;
        return 0;
    }

    /* Create receive Socket*/
    int sock_r;
    sock_r = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock_r<0) {
        cout << "ERROR creating socket! Exiting.. RETURN: " << sock_r << endl;
        return 0;
    }

    // Mode selection
    // Sender
    if (!mode) {


        struct sockaddr_in si_recv, si_send;
        socklen_t slen_send=sizeof(si_send);
        socklen_t slen_recv= sizeof(si_recv);
        unsigned char *rbuf = (unsigned char *) malloc(MAX_BUF_SIZE);

        memset((char *) &si_send, 0, sizeof(si_send));
        si_send.sin_family = AF_INET;
        si_send.sin_port = htons(DPORT);
        if (inet_aton(RECIEVER_IP , &si_send.sin_addr) == 0)
        {
            fprintf(stderr, "inet_aton() failed\n");
            exit(1);
        }
        memset((char *) &si_recv, 0, sizeof(si_recv));
        si_recv.sin_family = AF_INET;
        si_recv.sin_port = htons(DPORT);
        si_recv.sin_addr.s_addr = htonl(INADDR_ANY); // Or SENDER_IP

        uint8_t *pkt_buf;
        size_t pkt_size = sizeof(struct payload);
        pkt_buf = (uint8_t *) malloc(pkt_size);
        memset(pkt_buf,0, pkt_size);
        // Forge default payload
        struct payload *payload;
        payload = (struct payload*) pkt_buf;
        struct timespec init_time;
        clock_gettime(CLOCK_MONOTONIC, &init_time);
        payload->timestamp = init_time;
        payload->sequenceNum = 1;
        payload->isAck = 0;

        // Set Socket Timeout
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = timeout_usec;
        if (setsockopt(sock_r, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
            cout << "Error at setting Recv TIMEOUT !" << endl;
            return 0;
        }

        if( bind(sock_r , (struct sockaddr*)&si_recv, sizeof(si_recv) ) < 0) {
            cout << "Error Binding Socket ... " << endl;
            exit(0);
        }

        // Additional Params
        int current = 1, current_rt = 0;
        timespec stop;
        timespec start;
        timespec flow_start;
        timespec recv_time;
        timespec recv_ioctl;
        ssize_t recvlen;
        // Timeval test
        // Test ioctl
        struct timeval ack_t;
        int error;

        while (current <= num_pkts) {

            while (current_rt <= max_rt) {

                // Send Packet
                //if (current_rt == 0) {
                //    clock_gettime(CLOCK_REALTIME, &flow_start);
                //}
                //clock_gettime(CLOCK_MONOTONIC, &start);
                //payload->timestamp = start;
                payload->sequenceNum = (uint64_t) current;
                if (sendto(sock_s, pkt_buf, pkt_size, 0, (struct sockaddr *)&si_send, sizeof(struct sockaddr)) < 0) {
                    cout << "ERROR at sending packet" << endl;
                    return 0;
                } else {
                    if (current_rt == 0) {
                        //clock_gettime(CLOCK_REALTIME, &flow_start);
                        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                    }
                    //clock_gettime(CLOCK_MONOTONIC, &stop);
                }

                // Recieve ACK
                recvlen = recvfrom(sock_r,pkt_buf, pkt_size, 0, (struct sockaddr *) &si_recv, &slen_recv);
                //error = ioctl(sock_r, SIOCGSTAMP, &ack_t);
                //TIMEVAL_TO_TIMESPEC(&ack_t, &recv_ioctl);
                if ( recvlen < 0) {
                    current_rt++;
                    if (current_rt > max_rt) {
                        //clock_gettime(CLOCK_REALTIME, &recv_time);
                        clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                        //timespec flow_latency = diff(flow_start, recv_time);
                        timespec flow_latency = diff(start, stop);
                        long flow_lat = diff_in_ns(start, stop);
                        // SEQ, NUM_RTS, TOTAL_LATENCY, FLOW_START, FLOW_END, RECEIVED
                        //cout << payload->sequenceNum << "," << current_rt-1 << "," << flow_latency.tv_sec + flow_latency.tv_nsec / 1e9 << "," << flow_start.tv_sec << "." << flow_start.tv_nsec << "," << recv_time.tv_sec << "." << recv_time.tv_nsec << "," << 0 << endl;
                        cout << payload->sequenceNum << "," << current_rt-1 << "," << flow_lat / 1e9 << "," << start.tv_sec << "." << start.tv_nsec << "," << stop.tv_sec << "." << stop.tv_nsec << "," << 0 << endl;
                        break;
                    }
                } else if ( recvlen > 0 ) {
                    if (payload->isAck == 1 && (int) payload->sequenceNum == current) {
                        //timespec flow_latency = diff(flow_start, recv_ioctl);
                        clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
                        //timespec flow_latency = diff(start, stop);
                        long flow_lat = diff_in_ns(start, stop);
                        // Report results
                        // SEQ, NUM_RTS, TOTAL_LATENCY, FLOW_START, FLOW_END, RECEIVED
                        //cout << payload->sequenceNum << "," << current_rt << "," << flow_latency.tv_sec + flow_latency.tv_nsec / 1e9 << "," << flow_start.tv_sec << "." << flow_start.tv_nsec << "," << recv_ioctl.tv_sec << "." << recv_ioctl.tv_nsec << "," << 1 << endl;
                        cout << payload->sequenceNum << "," << current_rt << "," << flow_lat / 1e9 << "," << start.tv_sec << "." << start.tv_nsec << "," << stop.tv_sec << "." << stop.tv_nsec << "," << 1 << endl;
                        break;
                    } else {
                        //current_rt++;
                    }

                }

            }
            current++;
            current_rt = 0;
            payload->sequenceNum += 1;
            payload->isAck = 0;
        }
        close(sock_r);
        close(sock_s);

    // Receiver
    } else {

        // Create default payload
        struct sockaddr_in si_recv, si_send;
        socklen_t slen_send = sizeof(si_send);
        socklen_t slen_recv = sizeof(si_recv);
        ssize_t recv_len;
        int current = 1, before = 1, current_rt = 0;
        unsigned char *rbuf = (unsigned char *) malloc(MAX_BUF_SIZE);

        memset((char *) &si_recv, 0, sizeof(si_recv));
        memset((char *) &si_send, 0, sizeof(si_send));

        si_recv.sin_family = AF_INET;
        si_recv.sin_port = htons(DPORT);
        si_recv.sin_addr.s_addr = htonl(INADDR_ANY); // Or SENDER_IP

        si_send.sin_family = AF_INET;
        si_send.sin_port = htons(DPORT);
        if (inet_aton(SENDER_IP , &si_send.sin_addr) == 0)
        {
            fprintf(stderr, "inet_aton() failed\n");
            exit(1);
        }
        if( bind(sock_s , (struct sockaddr*)&si_recv, sizeof(si_recv) ) < 0) {
            cout << "Error Binding Socket ... " << endl;
            exit(0);
        }

        uint8_t *pkt_buf;
        size_t pkt_size = sizeof(struct payload);
        pkt_buf = (uint8_t *) malloc(pkt_size);
        memset(pkt_buf,0, pkt_size);
        struct payload *pay;
        pay = (struct payload*) pkt_buf;
        struct timespec init_time;
        clock_gettime(CLOCK_REALTIME, &init_time);
        pay->timestamp = init_time;
        pay->sequenceNum = 1;
        pay->isAck = 1;

        // Time params
        timespec recv_time;
        timespec start;
        timespec stop;
        // Test ioctl
        struct timeval recv_t;
        int error;

        while (true) {
            recv_len = recvfrom(sock_s, pkt_buf, pkt_size, 0, (struct sockaddr *) &si_recv, &slen_recv);
            error = ioctl(sock_s, SIOCGSTAMP, &recv_t);
            TIMEVAL_TO_TIMESPEC(&recv_t, &recv_time);
            if (recv_len < 0) {
                cout << "Error receiving Packet !" << endl;
            } else {
                current = (int) pay->sequenceNum;
                //cout << "SEQ: " << pay->sequenceNum << endl;
                if (pay->isAck == 0) {
                    pay->isAck = 1;
                    // Send ACK back to Sender
                    //clock_gettime(CLOCK_REALTIME, &start);
                    //pay->timestamp = start;
                    if (sendto(sock_r, pkt_buf, pkt_size, 0, (struct sockaddr *)&si_send, sizeof(struct sockaddr)) < 0) {
                        cout << "ERROR at sending packet" << endl;
                        return 0;
                    } else {
                        //cout << "Sent ACK for packet SEQ NUM: " << pay->sequenceNum << " CURRENT_RT : " << current_rt << endl;
                    }
                }
            }
            memset(pkt_buf,0, pkt_size);
        }
        close(sock_r);
        close(sock_s);
    }
}
