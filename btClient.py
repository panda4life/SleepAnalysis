# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:17:14 2014

@author: James Ahad
"""

import sys
import bluetooth as bt

print("performing inquiry...")

nearby_devices = bt.discover_devices(lookup_names = True)

print("found %d devices" % len(nearby_devices))

#initialize parameters
foundHypnolarm = False
hypno_service = None
for addr, name in nearby_devices:
    #print("  %s - %s" % (addr, name))
    services = bt.find_service(address=addr)
    if len(services) > 0:
        print("found %d services on %s" % (len(services), sys.argv[1]))
        print()
    else:
        print("no services found")
    for svc in services:
        if(svc["name"] == 'Hypnolarm: Alarm'):
            foundHypnolarm = True
            hypno_service = svc
            print("Found Hypnolarm at Bluetooth Address: %s" % addr)
            #print("Service Name: %s"    % svc["name"])
            #print("    Host:        %s" % svc["host"])
            #print("    Description: %s" % svc["description"])
            #print("    Provided By: %s" % svc["provider"])
            #print("    Protocol:    %s" % svc["protocol"])
            #print("    channel/PSM: %s" % svc["port"])
            #print("    svc classes: %s "% svc["service-classes"])
            #print("    profiles:    %s "% svc["profiles"])
            #print("    service id:  %s "% svc["service-id"])
            print()
            break

print("connecting to \"%s\" on %s" % ((hypno_service["host"], hypno_service["port"])))

# Create the client socket
sock=bt.BluetoothSocket( bt.RFCOMM )
sock.connect((hypno_service["host"], hypno_service["port"]))

print("connected.  type stuff")
while True:
    data = input()
    if len(data) == 0: break
    sock.send(data)

sock.close()