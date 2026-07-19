import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML Template with Embedded CSS styling (Shadows, layout, and responsiveness)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AI Customer Churn Prediction</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:'Poppins',sans-serif;
}


body{

    min-height:100vh;
    background:
    linear-gradient(135deg,#667eea,#764ba2);

    display:flex;
    justify-content:center;
    align-items:center;

    padding:30px;

}


.dashboard{


    width:100%;
    max-width:1100px;

    background:rgba(255,255,255,0.15);

    backdrop-filter:blur(20px);

    border-radius:25px;

    padding:35px;

    box-shadow:
    0 25px 50px rgba(0,0,0,0.25);

    color:white;

}



.header{

    text-align:center;
    margin-bottom:35px;

}


.header h1{

    font-size:38px;
    font-weight:700;

}


.header p{

    margin-top:10px;
    opacity:.8;

}



.form-grid{


display:grid;

grid-template-columns:repeat(3,1fr);

gap:22px;


}



.input-box{


background:white;

padding:15px;

border-radius:15px;

color:#333;


}



.input-box label{


font-size:14px;

font-weight:600;

display:block;

margin-bottom:8px;


}



input,select{


width:100%;

border:none;

outline:none;

font-size:15px;

padding:10px;

border-radius:10px;

background:#f4f6ff;


}



button{


margin-top:30px;

width:100%;

padding:15px;

border:none;

border-radius:15px;

background:#ff6b6b;

color:white;

font-size:18px;

font-weight:600;

cursor:pointer;


transition:.3s;


}



button:hover{


transform:translateY(-3px);

background:#ff4757;


}




.result{


margin-top:30px;

padding:25px;

border-radius:18px;

text-align:center;

font-size:22px;

font-weight:600;

display:none;


}



.churn{


background:#ffe0e0;

color:#c0392b;


}



.safe{


background:#d4ffd8;

color:#1e8449;


}



.progress{


height:15px;

background:#ddd;

border-radius:20px;

overflow:hidden;

margin-top:15px;


}



.progress-bar{


height:100%;

width:0%;

transition:1s;


}



@media(max-width:900px){


.form-grid{

grid-template-columns:1fr 1fr;

}

}



@media(max-width:600px){


.form-grid{

grid-template-columns:1fr;

}


.header h1{

font-size:28px;

}


}



</style>


</head>



<body>


<div class="dashboard">


<div class="header">


<h1>🏦 Customer Churn Analytics Prediction</h1>

<p>Machine Learning Powered Customer Retention Analytics</p>


</div>



<form id="prediction-form">


<div class="form-grid">



<div class="input-box">

<label>Credit Score</label>

<input type="number" name="credit_score" value="650">

</div>



<div class="input-box">

<label>Country</label>

<select name="country">

<option value="0">France</option>

<option value="1">Germany</option>

<option value="2">Spain</option>

</select>

</div>



<div class="input-box">

<label>Gender</label>

<select name="gender">

<option value="1">Male</option>

<option value="0">Female</option>

</select>

</div>




<div class="input-box">

<label>Age</label>

<input type="number" name="age" value="35">

</div>



<div class="input-box">

<label>Tenure</label>

<input type="number" name="tenure" value="5">

</div>




<div class="input-box">

<label>Account Balance</label>

<input type="number" name="balance" value="50000">

</div>




<div class="input-box">

<label>Products</label>

<input type="number" name="products_number" value="1">

</div>




<div class="input-box">

<label>Credit Card</label>

<select name="credit_card">

<option value="1">Yes</option>

<option value="0">No</option>

</select>

</div>



<div class="input-box">

<label>Active Member</label>

<select name="active_member">

<option value="1">Yes</option>

<option value="0">No</option>

</select>

</div>




<div class="input-box">

<label>Estimated Salary</label>

<input type="number" name="estimated_salary" value="60000">

</div>



</div>



<button>
🔍 Analyze Customer Risk
</button>


</form>




<div id="result" class="result">


</div>


</div>




<script>


document.getElementById("prediction-form")
.addEventListener("submit",async function(e){


e.preventDefault();


let form=new FormData(this);


let data={};


form.forEach((v,k)=>{


data[k]=Number(v);


});



let result=document.getElementById("result");



let response=await fetch("/predict",{


method:"POST",

headers:{

"Content-Type":"application/json"

},


body:JSON.stringify(data)


});



let output=await response.json();



result.style.display="block";



let probability=(output.probability*100).toFixed(2);



if(output.prediction==1){


result.className="result churn";


result.innerHTML=

`
⚠️ High Churn Probability

<h2>${probability}%</h2>

<div class="progress">

<div class="progress-bar" 
style="width:${probability}%;background:#e74c3c">

</div>

</div>

`;



}

else{


result.className="result safe";


result.innerHTML=

`
✅ Customer Likely To Stay

<h2>${probability}%</h2>

<div class="progress">

<div class="progress-bar"
style="width:${probability}%;background:#27ae60">

</div>

</div>

`;



}


});


</script>


</body>

</html>
"""
