## Start Postgraphql Command

export $(cat settings/.env | xargs)

With jwtToken
```
postgraphile -c $DB_CONNECTION_STRING\
  --schema api\
  --watch --token api.jwt_token \
  --secret 'keyboard_kitten' --port $POSTGRAPHQL_PORT

```


## Create User

```
mutation{registerUser(
  input:{phone:"234",
    		firstName:"name1",
    		lastName:"firstname1",
  			password:"admin",	roleId:3}
){
  user{nodeId,
  id,
  phone,
  firstName,
  lastName,
  createdate}

}}
```

## Get code


```
mutation{getCode(input:{inputPhone:"18616721470"}){

  string

}}
```

## VERIFY CodeGen


```
mutation{verifyCode
  (input:
  	{inputPhone:"18616721470",
			inputCode:32311}
	)
 {string}
}

```

## authenticate

mutation{authenticate
  (input:{
    inputPhone:"234",
    inputPassword: "admin"
  }
  )
  {
    jwtToken
  }
}
