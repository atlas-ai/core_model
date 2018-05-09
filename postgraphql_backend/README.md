## Start Postgraphql Command

SET ENV VARIABLES

```
export $(cat settings/.env | xargs)
```


RUN POSTRAPHILE USING jwtToken
```
postgraphile -c $DB_CONNECTION_STRING\
  --schema api\
  --watch --token $POSTGRAPHILE_SECRET \
  --secret 'keyboard_kitten' --port $POSTGRAPHILE_PORT
```



## API information

Parameters :
SMS_API_ACCOUNT : Atlas 3rd party SMS api account
SMS_API_PASSWORD : Atlas 3rd party SMS api password


SMS_PHONE : SMS recipient phone number
SMS_CONTENT : SMS content

Account
```
http://mt.10690404.com/receive.do?Account=SMS_API_ACCOUNT&Password=SMS_API_PASSWORD&Fmt=json
```

MESSAGE

```
http://mt.10690404.com/send.do?Account=SMS_API_ACCOUNT&Password=SMS_API_PASSWORD&Mobile=18616721479&Content=test&Exno=0&Fmt=json
```

ACCOUNT BALANCE
http://mt.10690404.com/getUser.do?Account=SMS_API_ACCOUNT&Password=SMS_API_PASSWORD&Fmt=json
