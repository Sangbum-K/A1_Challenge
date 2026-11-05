"""Read interface traffic and ARP-probe only the three known sensor addresses.
Requires CAP_NET_RAW. Does not change interfaces, routes, or sensor settings.
"""
import json
import select
import socket
import struct
import time
from pathlib import Path

def main():
    interfaces={'enp4s0':'192.168.1.100','enp0s31f6':'192.168.1.102','enp5s0':'192.168.137.2'}
    targets=['192.168.1.51','192.168.1.201','192.168.53.100']
    sockets={};found=set()
    try:
        for iface,ip in interfaces.items():
            s=socket.socket(socket.AF_PACKET,socket.SOCK_RAW,socket.htons(3));s.bind((iface,0));s.setblocking(False);sockets[s]=iface
            mac=bytes.fromhex((Path('/sys/class/net')/iface/'address').read_text().strip().replace(':',''))
            for target in targets:
                arp=struct.pack('!HHBBH',1,0x0800,6,4,1)+mac+socket.inet_aton(ip)+bytes(6)+socket.inet_aton(target)
                s.send(bytes([255])*6+mac+struct.pack('!H',0x0806)+arp)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            ready,_,_=select.select(list(sockets),[],[],.2)
            for s in ready:
                frame=s.recv(65535)
                if len(frame)<42:continue
                kind=struct.unpack('!H',frame[12:14])[0]
                if kind==0x0806:
                    sender=socket.inet_ntoa(frame[28:32]);op=struct.unpack('!H',frame[20:22])[0]
                    if sender in targets:found.add((sockets[s],'ARP',sender,op,frame[22:28].hex(':')))
                elif kind==0x0800:
                    sender=socket.inet_ntoa(frame[26:30])
                    if sender in targets:found.add((sockets[s],'IPv4',sender,frame[23],frame[6:12].hex(':')))
        print(json.dumps(sorted(found),indent=2))
    finally:
        for s in sockets:s.close()

if __name__=='__main__':main()
