var xs = [];
	var ys = [];
	document.getElementById('x').value = 1;

	const model = tf.sequential();

	model.add(tf.layers.dense({units : 128 , inputShape : [1]}));
	model.add(tf.layers.dense({units : 128 , inputShape : [128],activation: "sigmoid"}));
	model.add(tf.layers.dense({units : 1 , inputShape : [128]}));

	const optimizer = tf.train.adam(0.05);

	model.compile({optimizer: optimizer, loss: 'meanSquaredError'});

	document.getElementById('append').onclick = function(){

	var x = document.getElementById('x').value;
	var y = document.getElementById('y').value;

	xs.push(x);
	ys.push(y);

	var tfxs = tf.tensor1d(xs);
	var tfys = tf.tensor1d(ys);


	document.getElementById('x').value = parseInt(x) + 1;

	model.fit(tfxs , tfys , {epochs : 200}).then(() => {
		bestfit = model.predict(tf.tensor(xs, [xs.length, 1])).dataSync();

		var ctx = document.getElementById("myChart").getContext('2d'); // begin chart

	            // Chart data and settings:
	            var myChart = new Chart(ctx, {
	                type: 'line',
	                options: {scales:{yAxes: [{ticks: {beginAtZero: true}}]}},
	                data: {
	                    labels: xs,
	                    datasets: [
	                    {
	                        label: 'Original Data',
	                        data: ys,
	                        borderWidth: 1,
	                    } ,{
	                            label: 'Best Fit line',
	                            data: bestfit,
	                            borderWidth: 1,
	                            borderColor: '#FF0000',
	                            backgroundColor: 'rgba(1,1,1,0)'
	                        },
	                    ]
	                },
	            });
          })

	// console.log(xs,ys);

	}