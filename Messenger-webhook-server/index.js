'use strict';
const PAGE_ACCESS_TOKEN = process.env.PAGE_ACCESS_TOKEN;
// Imports dependencies and set up http server
const 
  request = require('request'),
  express = require('express'),
  body_parser = require('body-parser'),
  app = express().use(body_parser.json()); // creates express http server

// Sets server port and logs message on success
app.listen(process.env.PORT || 1337, () => console.log('webhook is listening'));

// Accepts POST requests at /webhook endpoint
app.post('/webhook', (req, res) => {  

  // Parse the request body from the POST
  let body = req.body;

  // Check the webhook event is from a Page subscription
  if (body.object === 'page') {

    body.entry.forEach(function(entry) {

      // Gets the body of the webhook event
      let webhook_event = entry.messaging[0];
      console.log(webhook_event);


      // Get the sender PSID
      let sender_psid = webhook_event.sender.id;
      console.log('Sender ID: ' + sender_psid);

      // Check if the event is a message or postback and
      // pass the event to the appropriate handler function
      if (webhook_event.message) {
        console.log("webhook_event")
        handleMessage(sender_psid, webhook_event.message);        
      } else if (webhook_event.postback) {
        
        handlePostback(sender_psid, webhook_event.postback);
      }
      
    });
    // Return a '200 OK' response to all events
    res.status(200).send('EVENT_RECEIVED');

  } else {
    // Return a '404 Not Found' if event is not from a page subscription
    res.sendStatus(404);
  }

});

// Accepts GET requests at the /webhook endpoint
app.get('/webhook', (req, res) => {
  
  /** UPDATE YOUR VERIFY TOKEN **/
  const VERIFY_TOKEN = "1234";
  
  // Parse params from the webhook verification request
  let mode = req.query['hub.mode'];
  let token = req.query['hub.verify_token'];
  let challenge = req.query['hub.challenge'];
    
  // Check if a token and mode were sent
  if (mode && token) {
  
    // Check the mode and token sent are correct
    if (mode === 'subscribe' && token === VERIFY_TOKEN) {
      
      // Respond with 200 OK and challenge token from the request
      console.log('WEBHOOK_VERIFIED');
      res.status(200).send(challenge);
    
    } else {
      // Responds with '403 Forbidden' if verify tokens do not match
      res.sendStatus(403);      
    }
  }
});

function handleMessage(sender_psid, received_message) {
  let response;
  // Checks if the message contains text
  if (received_message.text) {    
    // Create the payload for a basic text message, which
    // will be added to the body of our request to the Send API
    

    request({
      url: " https://curvy-robin-24.localtunnel.me/transtext",
      method: "POST",
      json: true,   // <--Very important!!!
      body: received_message.text
  }, function (error, response, body){
      if (error){
        console.log(error)
      }
      console.log(body)
      response = {
        "text": body.results
      }
      let request_body = {
        "recipient": {
          "id": sender_psid
        },
        "message": response
      }
      // Send the HTTP request to the Messenger Platform
      request({
        "uri": "https://graph.facebook.com/v2.6/me/messages",
        "qs": { "access_token": "EAASBgxQETo8BAPcq0AK67YKwE4y3MTrpH7ZB5ZBBKq75CsWZB3h2CXkqDfOFR6jmIgWZAehNTBjaIbkSQNqIaipmF2rSGkQPihzh9mwDsvv9w1imkNTpZBonKZA7vt0Ug59e1leRyD4M2GCrqHmoY5ZC2XoS7X0bS6RyR4FUxNW9wZDZD" },
        "method": "POST",
        "json": request_body
      }, (err, res, body) => {
        if (!err) {
          console.log('message sent!')
        } else {
          console.error("Unable to send message:" + err);
        }
      }); 
    });

  } else if (received_message.attachments) {
    console.log("got attachment...")
    // Get the URL of the message attachment
    let img = received_message.attachments;
    request({
      url: " https://curvy-robin-24.localtunnel.me/predict",
      method: "POST",
      json: true,   // <--Very important!!!
      body: img
  }, function (error, response, body){
      console.log(body)
      console.log(typeof(body.result))
      console.log(response.body);
      console.log(sender_psid);

      if (error){
        console.log(error)
      }
      response = {
        "text": body.result
      }
      let request_body = {
        "recipient": {
          "id": sender_psid
        },
        "message": response
      }
      // Send the HTTP request to the Messenger Platform
      request({
        "uri": "https://graph.facebook.com/v2.6/me/messages",
        "qs": { "access_token": "EAASBgxQETo8BAPcq0AK67YKwE4y3MTrpH7ZB5ZBBKq75CsWZB3h2CXkqDfOFR6jmIgWZAehNTBjaIbkSQNqIaipmF2rSGkQPihzh9mwDsvv9w1imkNTpZBonKZA7vt0Ug59e1leRyD4M2GCrqHmoY5ZC2XoS7X0bS6RyR4FUxNW9wZDZD" },
        "method": "POST",
        "json": request_body
      }, (err, res, body) => {
        if (!err) {
          console.log('message sent!')
        } else {
          console.error("Unable to send message:" + err);
        }
      }); 
  });
  } 
  callSendAPI(sender_psid, response);
  // Send the response message
     
}

function handlePostback(sender_psid, received_postback) {
   let response;
  // Get the payload for the postback
  let payload = received_postback.payload;

  // Set the response based on the postback payload
  if (payload === 'yes') {
    response = { "text": "Thanks!" }
  } else if (payload === 'no') {
    response = { "text": "Oops, try sending another image." }
  }
  // Send the message to acknowledge the postback
  callSendAPI(sender_psid, response);
}


function callSendAPI(sender_psid, response) {
  // console.log(typeof(response))
  // Construct the message body
  let request_body = {
    "recipient": {
      "id": sender_psid
    },
    "message": response
  }
  // Send the HTTP request to the Messenger Platform
  request({
    "uri": "https://graph.facebook.com/v2.6/me/messages",
    "qs": { "access_token": "EAASBgxQETo8BAPcq0AK67YKwE4y3MTrpH7ZB5ZBBKq75CsWZB3h2CXkqDfOFR6jmIgWZAehNTBjaIbkSQNqIaipmF2rSGkQPihzh9mwDsvv9w1imkNTpZBonKZA7vt0Ug59e1leRyD4M2GCrqHmoY5ZC2XoS7X0bS6RyR4FUxNW9wZDZD" },
    "method": "POST",
    "json": request_body
  }, (err, res, body) => {
    if (!err) {
      console.log('message sent!')
    } else {
      console.error("Unable to send message:" + err);
    }
  }); 
}