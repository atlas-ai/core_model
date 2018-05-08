import requests
import re
import json

#r = requests.post('http://localhost:5003/graphql', data = {'key':'value'})


#headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhIjoiYWEiLCJiIjoiYmIiLCJjIjoiY2MiLCJpYXQiOjE1MjIxMTYzMzAsImV4cCI6MTUyMjIwMjczMCwiYXVkIjoicG9zdGdyYXBoaWxlIiwiaXNzIjoicG9zdGdyYXBoaWxlIn0.UiXBzMrY1JfQKgbbcMA-6CrQGVgVz4c9w2IbPkTrsfA"}
headers=''

def run_query(query,headers=''): # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.post('http://localhost:5003/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))





def get_jwt(user_id):
    query="""
    mutation{{
      generateJwt(input:{{userId:{0}}}) {{
        clientMutationId
        jwtToken
      }}
    }}
    """.format(user_id)
    result=run_query(query)
    return (result['data']['generateJwt']['jwtToken'])

def set_header(user_id):
    jwtToken=get_jwt(user_id)
    header = {"Authorization": "Bearer {0}".format(jwtToken)}
    return header



def read_jwt(headers=''):
    query="""query{
    readjwt
    }"""

    result=run_query(query,headers)
    print(result)

def read_jwt_role(headers=''):
    query="""query{
readjwtRole
}"""

    result=run_query(query,headers)
    print(result)


def select_all(query,headers):
    result=run_query(query,headers)['data']
    return result[next(iter(result))]['nodes']


#query="query{ allUsers{nodes{id,phone}}}"


#request = requests.post('http://localhost:5003/graphql', json={'query': query})
#print (request.json())
phone=444
queries=[
{"queryName":"getUserWithoutJWT","query":"query{ allUsers{nodes{id,phone}}}","result": """{"allUsers": null}"""},
{"queryName": "getSMSCode","query":"""mutation{{getCode(input:{{inputPhone:"{}"}}){{string}}}}""".format(phone),"result":"""{"getCode": {"string": "[0-9]{6}"}}"""},
#{"queryName": "verifyCode","query":"""mutation{verifyCode(input:{inputPhone:"2",inputCode:CODE} ){string}}""","result":"""{"getCode": {"string": "[0-9]{6}"}}"""},
{"queryName": "verifyCode","query":"""mutation{{verifyCode(input:{{inputPhone:"{}",inputCode:1111}} ){{string}}}}""".format(phone),"result":"""{"verifyCode": {"string": "CODE VALID"}}"""},
{"queryName": "registerUser","query":"""mutation{{registerUser(input:{{phone:"{}",firstName:"A",lastName:"A",password:"admin",roleId:4 }}) {{user {{ phone}}}}}}""".format(phone),"result":"""{{"registerUser": {{"user": {{"phone": "{}"}}}}}}""".format(phone)},
{"queryName": "authenticationFail","query":"""mutation{{authenticate(input:{{inputPhone:"{}",inputPassword:"admin"}}){{jwtToken}} }}""".format(9999999),"result":"a"},
{"queryName": "authentication","query":"""mutation{{authenticate(input:{{inputPhone:"{}",inputPassword:"admin"}}){{jwtToken}} }}""".format(phone),"result":"a"},
{"queryName":"getUserWithJWT","query":"query{ allUsers{nodes{id,phone}}}","result": """{"allUsers": null}"""},




    ]

# TEST3
def run_tests():
    headers=''
    for query in queries:
        result=run_query(query['query'],headers)
        to_string=json.dumps(result['data'])
        print(result)
        #print(to_string)
        #print(query['result'])
        #print (re.match(query['result'], to_string))

        if query['queryName']=='getSMSCode':
            m=re.search("[0-9]{6}", to_string) #""""{"getCode": {"string": "333444"}}""")
            if m:
                code=m.group(0)
                query_With_New_Code =queries[2]['query'].replace('1111',code)
                queries[2]['query']=query_With_New_Code
            else:
                print("no code")

        if query['queryName']=='authentication':
            json_load=json.loads(to_string)
            jwtToken= json_load['authenticate']['jwtToken']
            headers= {"Authorization": "Bearer {0}".format(jwtToken)}
            #print(header)





run_tests()
