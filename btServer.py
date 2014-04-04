# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:32:43 2014

@author: James Ahad
"""

# file: rfcomm-server.py
# auth: Albert Huang <albert@csail.mit.edu>
# desc: simple demonstration of a server application that uses RFCOMM sockets
#
# $Id: rfcomm-server.py 518 2007-08-10 07:20:07Z albert $

def processCommand(packet):
    # Decompose the packet
    packet = packet.strip()
    columns = packet.split(',')
    cmd = int(columns[0])
    if(len(columns) > 1):
        data = columns[1:]

    # Process the packet
    # 0 means pass, or do nothing
    if(cmd == 0):
        return
    # 1 means data input
    elif(cmd == 1):
        processData(data)
    # anything else is an invalid command
    # invalid commands get error message printed to log
    # and get treated as a pass
    else:
        print('Invalid Command. Command is %i' % cmd)
        return




import bluetooth as bt

server_sock=bt.BluetoothSocket( bt.RFCOMM )
server_sock.bind(("",bt.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

bt.advertise_service( server_sock, "Hypnolarm: Alarm",
                   service_id = uuid,
                   service_classes = [ uuid, bt.SERIAL_PORT_CLASS ],
                   profiles = [ bt.SERIAL_PORT_PROFILE ],
#                   protocols = [ OBEX_UUID ]
                    )

print("Waiting for connection from Bio-Amp on port %d" % port)
client_sock, client_info = server_sock.accept()

print("Accepted connection from ", client_info)

try:
    while True:
        data = client_sock.recv(1024)
        if len(data) == 0: continue
        processCommand(data)
except IOError:
    pass

print("disconnected")

client_sock.close()
server_sock.close()
print("all done")





