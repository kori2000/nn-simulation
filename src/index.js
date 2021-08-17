const { Timer } = require('timer-node')
const cliProgress = require('cli-progress')
const _colors = require('colors')

const dataset = require('./ds_source').dataset;
const dn = require('dannjs')

require('dotenv').config()

const Dann = dn.dann
const nn = new Dann(4, 4)

// nn.addHiddenLayer(16, 'leakyReLU')
// nn.outputActivation('sigmoid')

nn.addHiddenLayer(16,'tanH')
nn.outputActivation('sigmoid')

nn.makeWeights()
// nn.setLossFunction('mael')
nn.lr = 0.1


nn.log()

const ASOMN = [0, 1, 1, 0]
const EXPCT = [0, 1, 1, 1]

const EPOCH = process.env.EPOCH || 70000
let enu = 0

console.log(`NeuralNetwork training with EPOCH size [${EPOCH}] started...☕\n`)

const timer = new Timer()
timer.start()
const bar = new cliProgress.SingleBar({
  format: ' {percentage}% |' + _colors.magenta('{bar}') + '| Epoch {value} | Loss: {nnloss}',
  barCompleteChar: '\u2588',
  barIncompleteChar: '\u2591',
  hideCursor: true
}, cliProgress.Presets.shades_classic)
bar.start(EPOCH, 0, { nnloss: "N/A" })

while(++enu <= EPOCH) {  
  let sum = 0
  for (data of dataset) {
    nn.train(data.in, data.ta)
    sum += nn.loss
  }
  let avgLoss = sum/dataset.length
  bar.update(enu, {
    nnloss: avgLoss
  })
  
}

timer.pause()
bar.stop()

console.log(`\nDone training. Took ${timer.format('%m minutes, %s seconds, %ms ms')}\n`)

let prediction = nn.feedForward(ASOMN, {log: true, decimals: 18})
prediction = prediction.map(r => Math.round(r, 2))

console.log("ASSUMPTION ", ASOMN)
console.log("--------------------------")
console.log("EXPECTED   ", EXPCT)
console.log("PREDICTION ", prediction)
console.log("");

// let equals = (a, b) => JSON.stringify(a) === JSON.stringify(b)
// console.log(equals(EXPCT, prediction))

// console.log(nn.toJSON())

// console.log(nn.toFunction())