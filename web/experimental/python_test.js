const spawn = require('child_process').spawn;
const pythonProcess = spawn('python', ['test.py', 'hello']);

pythonProcess.stdout.on('data', (data) => {
  console.log(data.toString());
});
