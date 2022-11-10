import logo from './logo.svg'
import './App.css'
import React, { useState } from 'react'
import axios from 'axios'

function App () {
  const [champ1, setChamp1] = useState('')
  const [champ2, setChamp2] = useState('')
  const [champ3, setChamp3] = useState('')
  const [champ4, setChamp4] = useState('')
  const [champ5, setChamp5] = useState('')
  const [champ6, setChamp6] = useState('')
  const [champ7, setChamp7] = useState('')
  const [champ8, setChamp8] = useState('')
  const [champ9, setChamp9] = useState('')
  const [champ10, setChamp10] = useState('')
  const [result, setResult] = useState('')
  const setChampChange = (num, e) => {
    e.preventDefault()
    switch (num) {
      case 1:
        setChamp1(e.target.value)
        break
      case 2:
        setChamp2(e.target.value)
        break
      case 3:
        setChamp3(e.target.value)
        break
      case 4:
        setChamp4(e.target.value)
        break
      case 5:
        setChamp5(e.target.value)
        break
      case 6:
        setChamp6(e.target.value)
        break
      case 7:
        setChamp7(e.target.value)
        break
      case 8:
        setChamp8(e.target.value)
        break
      case 9:
        setChamp9(e.target.value)
        break
      case 10:
        setChamp10(e.target.value)
        break
      default:
        break
    }
  }
  const handleSubmit = e => {
    e.preventDefault()
    console.log(champ1)
    console.log(champ2)
    console.log(champ3)
    console.log(champ4)
    console.log(champ5)
    console.log(champ6)
    console.log(champ7)
    console.log(champ8)
    console.log(champ9)
    console.log(champ10)
    // make axios call

    axios
      .post('http://127.0.0.1:5000/getPrediction', {
        data: [
          champ1,
          champ2,
          champ3,
          champ4,
          champ5,
          champ6,
          champ7,
          champ8,
          champ9,
          champ10
        ]
      })
      .then(response => {
        const res = response.data
        setResult(res.prediction)
      })
      .catch(error => {
        if (error.response) {
          console.log(error.response)
          console.log(error.response.status)
          console.log(error.response.headers)
        }
      })

    /*fetch(
      `http://127.0.0.1:5000/getPrediction`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin':  "http://127.0.0.1:5000",
          'Access-Control-Allow-Methods': "POST",
          'Access-Control-Allow-Headers': "Content-Type, Authorization"
          
        },
        body: [champ1, champ2, champ3, champ4, champ5, champ6, champ7, champ8, champ9, champ10]
      }
    )
      .then(response => response.json())
      .catch(error => console.log(error))
      */
  }
  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Welcome to LOL Outcome Predictor!</h1>
        <p>
          <form
            onSubmit={e => {
              handleSubmit(e)
            }}
          >
            <ul>
              Team 1:
              <li>
                <input
                  type='text'
                  name='summ1'
                  value={champ1}
                  onChange={e => setChampChange(1, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ2'
                  value={champ2}
                  onChange={e => setChampChange(2, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ3'
                  value={champ3}
                  onChange={e => setChampChange(3, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ4'
                  value={champ4}
                  onChange={e => setChampChange(4, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ5'
                  value={champ5}
                  onChange={e => setChampChange(5, e)}
                />
              </li>
              Team 2:
              <li>
                <input
                  type='text'
                  name='summ6'
                  value={champ6}
                  onChange={e => setChampChange(6, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ7'
                  value={champ7}
                  onChange={e => setChampChange(7, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ8'
                  value={champ8}
                  onChange={e => setChampChange(8, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ9'
                  value={champ9}
                  onChange={e => setChampChange(9, e)}
                />
              </li>
              <li>
                <input
                  type='text'
                  name='summ10'
                  value={champ10}
                  onChange={e => setChampChange(10, e)}
                />
              </li>
              <input type='submit' value='Submit' />
            </ul>
          </form>
        </p>
        {result}
      </header>
    </div>
  )
}

export default App
