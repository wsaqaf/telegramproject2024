#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
from types import FunctionType
import os
import sys
import pandas as pd
import re
from pprint import pprint

import configparser
from telethon import TelegramClient, events, sync,functions, types, utils
from telethon.errors import SessionPasswordNeededError
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.functions.messages import GetMessagesReactionsRequest
from telethon.tl.types import InputPeerChannel

import telethon

api_id = 0000000
api_hash = ''

after_date = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone(offset=datetime.timedelta()))
before_date = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone(offset=datetime.timedelta()))

limit=10000000000

if __name__ == "__main__":
    if (len(sys.argv)>1):
        channel=sys.argv[1]
        if (len(sys.argv)>2):
            limit=int(float(sys.argv[2]))
    else:
        print("Missing channel name")
        sys.exit(1)

print ("###############\n"+channel+"\n###############")

client = TelegramClient('session_name', api_id, api_hash)
#       0       1     2       3       4           5       6         7           8                 9         10          11            12         13          14         15           17		
#header=['id','url','sender','date','forwards','views','replies','is_reply','reply_to_msg_id','is_forward','fwd_from','via_bot','has_photo','web_preview','mentions','hashtags','other_content
header=['id','url','sender','date','forwards','views','is_reply','reply_to_msg_id','is_forward','fwd_from','via_bot','has_photo','web_preview','mentions','hashtags','other_content']
emojis = ['👍','👎','❤','🔥','🥰','👏','😁','🤯','😱','🤬','😢','🤩','🙏']
header=header+['total_reactions']+emojis+['other_reactions','message']
#consider in the future adding 'reactions as a set of variables',
doc_types=["audio","voice","video","video_note","gif","sticker","contact","game","geo","invoice","poll","venue","dice"]
index=0
last_msg=''

#print(header)
#sys.exit(1)

fname=channel+".csv"

if (os.path.isfile(fname)):
    df = pd.read_csv(fname, sep=';', escapechar='\\')
    if (df.shape[0]>0):
        last_msg=df.iloc[-1].id
if last_msg:
    print("Resuming from id: ["+ str(last_msg)+"]")
    id=last_msg[last_msg.rindex('_')+1:]
    current_id=int(id)
else:
    current_id=0

async def main():
    global csv
    global index
    global channel
    global current_id
    global writer
    global after_date
    global before_date
    global limit
    global header
    global emojis
    global doc_types
    global api_id
    global api_hash

    async for message in client.iter_messages(channel, reverse = True, offset_date=after_date, min_id=current_id,limit=limit):
        if message.date>before_date:
            print ("Exiting loop since "+str(message.date)+" exceeds "+str(before_date))
            return
        line=[]
        index=index+1

        line.append(channel+"_"+str(message.id)) #id:0
        line.append("https://t.me/"+channel+"/"+str(message.id)) #url:1
        if message.post_author:
            line.append(message.post_author) #sender:2
        else:
            line.append(channel)
        line.append(str(message.date))  #date:3
        line.append(message.forwards)   #forwards: 4
        line.append(message.views)      #views:5
        if (message.is_reply):          #is_reply:7
            line.append("1")
            if (message.reply_to_msg_id):   #reply_to_msg_id:8
                line.append(channel+"_"+str(message.reply_to_msg_id))  #fwd_from:10
            else:
                line.append("")
        else:
            line.append("0")
            line.append("")
        if (message.fwd_from):        #is_forward:9
            if (message.fwd_from.from_id):
                try:
                    user=await client.get_entity(message.fwd_from.from_id)
                    line.append("1")
                    line.append(user.username)  #fwd_from:10
                except:
                    line.append("1")
                    line.append("<private>")
            else:
                line.append("0")
                line.append("")
        else:
            line.append("0")
            line.append("")
        if (message.via_bot):       #via_bot:11
            line.append("1")
        else:
            line.append("0")
        if message.photo:           #photo:12
            line.append("1")
        else:
            line.append("0")
        if message.web_preview:
            line.append(message.web_preview.url) #web_preview:13
        else:
            line.append("")
        valid_mentions=""
        if (message.message):
            mentions=re.findall("@([a-zA-Z0-9_]{1,100})",message.message)
            for mention in mentions:
                try:
                    if (1):
                        if (not valid_mentions):
                            valid_mentions='@'+mention.lower()
                        else:
                            valid_mentions=valid_mentions+' @'+mention.lower()
                except:
                    result = ""
        line.append(valid_mentions)
        valid_hashtags=""
        if (message.message):
            hashtags=re.findall("#([^\s\!\@\#\$\%\^\&\*\(\)\=\+\./\,\[\{\]\}\;\:\'\"\?><]+)",message.message)
            for hashtag in hashtags:
                if (len(hashtag)>2):
                    if (not valid_hashtags):
                        valid_hashtags='#'+hashtag
                    else:
                        valid_hashtags=valid_hashtags+' #'+hashtag
        line.append(valid_hashtags)
        docs=[]
        for msg_type in doc_types:
            if (getattr(message,msg_type)):
                docs.append(msg_type)
        if (docs):
            line.append(','.join(docs))           #other_content:16
        else:
            line.append("")

        updates =  await client(GetMessagesReactionsRequest(peer=channel, id=[message.id]))
        
        total_reactions = 0
        reaction_counts = {column: 0 for column in emojis}
        other_reactions = ""
        other_reactions_count= 0

        try:
        	reactions = updates.updates[0].reactions.results
        	for reaction_count in reactions:
        		reaction = reaction_count.reaction.emoticon
        		count = reaction_count.count
        		total_reactions += count
        		if reaction in emojis: reaction_counts[reaction]=count
        		else: 
        			other_reactions_count += count
        			other_reactions = other_reactions+f'{reaction}:{count},'
        except:
        	total_reactions = 0
        
        line.append(total_reactions)
        for k, v in reaction_counts.items(): line.append(str(v))
        if (other_reactions_count>0): line.append("["+str(other_reactions_count)+"] "+other_reactions[:-1])
        else: line.append("")
		
        line.append(message.message)                #message:17
        writer.writerow(line)
        print(str(index)+") Added: "+str(message.id)+" at "+str(message.date).split('+', 1)[0]+" ["+channel+"]")
if last_msg:
    myFile = open(fname, "a")
    writer = csv.writer(myFile,delimiter =';')
else:
    myFile = open(fname, "w")
    writer = csv.writer(myFile,delimiter =';')
    writer.writerow(header)

with client:
    client.loop.run_until_complete(main())

myFile.close()
print("Saved CSV file to "+fname)
