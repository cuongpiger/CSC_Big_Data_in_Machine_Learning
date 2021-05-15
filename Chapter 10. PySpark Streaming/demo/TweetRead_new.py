import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json


# Set up your credentials
# read key from files
file1 = open("key/access_secret.txt","r")  
file2 = open("key/access_token.txt","r")  
file3 = open("key/consumer_key.txt","r")
file4 = open("key/consumer_secret.txt","r")

# set up keys
consumer_key = file3.read()
consumer_secret = file4.read()
access_token = file2.read()
access_secret = file1.read()

# Create a TweetsListener, which will be responsible for the streaming itself.
class TweetsListener(StreamListener):

    def __init__(self, csocket):
        self.client_socket = csocket
    # on_data(): responsible for receiving data from the Twitter stream and sending it to a socket
    def on_data(self, data):
        try:
            msg = json.loads( data )
            print( msg['text'].encode('utf-8') )
            self.client_socket.send( msg['text'].encode('utf-8') )
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
    # on_error(): is used for receiving error messages.
    def on_error(self, status):
        print(status)
        return True

# sendData(): connect to Twitter streaming with c_socket object as a parameter 
def sendData(c_socket):
    # we authorize with our credentials and create a Stream object instance
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListener(c_socket))
    # pick every tweet containing the tag, ex: 'Covid 19'
    twitter_stream.filter(track=['Covid 19', 'COVID-19', 'Corona virus', 'Corona', 'Covid'])

if __name__ == "__main__":
    s = socket.socket()    # Create a socket object
    host = "127.0.0.1"     # Get local machine name
    port = 5555            # Reserve a port for your service.
    s.bind((host, port))   # Bind to the port

    print("Listening on port: %s" % str(port))

    s.listen(5)                 # Now wait for client connection.
    c, addr = s.accept()        # Establish connection with client.

    print( "Received request from: " + str( addr ) )    
    
    sendData(c)
