/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
// they are not used in the stroop code but may be useful to you

// All pages to be loaded
var pages = [
	"consent.html",
	"instructions/instruct-intro.html",
	"instructions/instruct-examples.html",
	"instructions/instruct-ready.html",
	"instructions/instruct-demo.html",
	"stage.html",
	"postquestionnaire.html",
];

psiTurk.preloadPages(pages);

var instructionPages = [ // add as a list as many pages as you like
	"instructions/instruct-intro.html",
];

var instructionPages2 = [ // add as a list as many pages as you like
	"instructions/instruct-ready.html",
];

let allSelectedEntries = [];

/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested 
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and 
* insert them into the document.
*
********************/

/********************
* VAR TEST       *
********************/
var stims
var stim
var example_stims
var example_stim
var modality

const csvUrls = [

];  // Add the CSV file paths that include the paths to the video stimuli and their associated labels.

function VARExperiment_Instructions() {
	example_stims = [
		["path to the demonstration video stimuli 1", "label"],
		["path to the demonstration video stimuli 2", "label"]
	] // Add sample paths for demonstration video stimuli and their associated labels.

	var next = function() {
		if (example_stims.length==0) {
			psiTurk.doInstructions(instructionPages2,function() { currentview = new EntriesSelection(csvUrls); });	
		}
		else {
			example_stim = example_stims.shift();

			// enable and display the startButton
			let startButton = d3.select("#startButton");
			startButton.attr("disabled", null);
			startButton.style("display", "inline");
			
			// disable and hide the submitButton
			var submitButton = document.getElementById("submitButton");
			submitButton.removeEventListener("click", handleButtonClick); 
			submitButton.style.display = "none";
			submitButton.disabled = true;

			// shuffle positions of options
			let radioItems = Array.from(document.querySelectorAll('.radio-item'));
			radioItems.sort(() => Math.random() - 0.5);
			let container = document.querySelector('.radio-container');
			container.innerHTML = '';
			let column1 = document.createElement('div');
			column1.className = 'radio-column';
			let column2 = document.createElement('div');
			column2.className = 'radio-column';
			radioItems.forEach((item, index) => {
				if (index < 5) {
					column1.appendChild(item);
				} else {
					column2.appendChild(item);
				}
			});
			
			container.appendChild(column1);
			container.appendChild(column2);

			// set warnings
			var arrow1 = document.getElementById("text-with-arrow1");
			arrow1.style.display = "inline";
			var arrow2 = document.getElementById("text-with-arrow2");
    		arrow2.style.display = "none";
			var arrow3 = document.getElementById("text-with-arrow3");
    		arrow3.style.display = "none";

			var textContainer = document.getElementById("WARN");
			textContainer.innerHTML = "Wrong Selection!";
			textContainer.style.fontSize = "24px";  
			textContainer.style.fontWeight = "bold"; 
			textContainer.style.color = "orange";  
			textContainer.style.display = "none";    

			var textContainer = document.getElementById("WARN2");
			textContainer.innerHTML = "Please select one option before proceeding to the next page!";
			textContainer.style.fontSize = "24px";  
			textContainer.style.fontWeight = "bold"; 
			textContainer.style.color = "orange";  
			textContainer.style.display = "none";
			
			// if the startButton is clicked, show the video
			startButton.on("click", function() {
    			var arrow1= document.getElementById("text-with-arrow1");
    			arrow1.style.display = "none";

				show_video(example_stim[0]);
			});

		}
	};

	function handleButtonClick() {
		var checkedOption = document.querySelector('input[name="option"]:checked');
		if (checkedOption) {
			selectedOption = checkedOption.id 
			if (selectedOption != example_stim[1]){
				var textContainer = document.getElementById("WARN2"); 
				textContainer.style.display = "None";  
				var textContainer = document.getElementById("WARN");
				textContainer.style.display = "inline";  
	
			}
			else{
				// clear the option
				if (checkedOption) {
					checkedOption.checked = false;
				}
				remove_video();
				next();
			}
		}
		else{
			var textContainer = document.getElementById("WARN2"); 
			textContainer.style.display = "inline";  
		}

		var checkedOption = document.querySelector('input[name="option"]:checked');
		selectedOption = checkedOption.id

		if (selectedOption != example_stim[1]){
			var textContainer = document.getElementById("WARN");
			textContainer.style.display = "inline";  

		}
		else{
			// clear the option
			if (checkedOption) {
				checkedOption.checked = false;
			}
			remove_video();
			next();
		}
		
	}

	var show_video = function(videoPath) {
		// when the video is showed, disable and hide startButton
		let startButton = d3.select("#startButton");
		startButton.attr("disabled", true);
		startButton.style("display", "none");

		var video = d3.select("#stim")
		.append("video")
		.attr("id", "video")
		.attr("width", "398")
		.attr("height", "224")
		.attr("autoplay", true)
		.append("source")
		.attr("src", videoPath)
		.attr("type", "video/mp4");

		d3.select("#video").on("ended", function() {
			// when the video is end, enable and display submitButton
			d3.select(this).style("display", "none");
			var submitButton = document.getElementById("submitButton");
			submitButton.style.display = "inline";
			submitButton.disabled = false;
			submitButton.addEventListener("click", handleButtonClick); 

			var arrow2 = document.getElementById("text-with-arrow2");
			arrow2.style.display = "inline";
			var arrow3 = document.getElementById("text-with-arrow3");
			arrow3.style.display = "inline";
		});			
		
	
	};

	var remove_video = function() {
		d3.select("#video").remove();
	};

	psiTurk.showPage('instructions/instruct-examples.html');

	// Start the test
	next();
}


function EntriesSelection(urls, index = 0){
	if (index >= urls.length) {
		VARExperiment();
	}

	Papa.parse(urls[index], {
		download: true,
		complete: function(results) {

			let data = results.data;
			let categorizedData = {};

			data.forEach(item => {
				let path = item[0];
				let label = item[1];
				if (!categorizedData[label]) {
					categorizedData[label] = [];
				}
				categorizedData[label].push(path);
			});

			let selectedEntries = [];
			for (let label in categorizedData) {
				let entries = categorizedData[label];
				let shuffledEntries = entries.sort(() => 0.5 - Math.random());
				let selected = shuffledEntries.slice(0, 2); // select 2 samples for each action class
				selected.forEach(path => {
					selectedEntries.push({path: path, label: label});
				});
			}

			allSelectedEntries = allSelectedEntries.concat(selectedEntries);
			EntriesSelection(urls, index + 1);
		}
	});

};


function VARExperiment() {
	let stims = allSelectedEntries
	var next = function() {
		if (stims.length==0) {
			// disable and hide ten options
			let radioItems = Array.from(document.querySelectorAll('.radio-item'));
			radioItems.forEach(item => {
				item.style.display = "none"; 
				item.disabled = true;       
			});
			
			// disable and hide the submitButton
			var submitButton = document.getElementById("submitButton");
			submitButton.removeEventListener("click", handleButtonClick); 
			submitButton.style.display = "none";
			submitButton.disabled = true;

			currentview = new Questionnaire()
		}
		else {
			let randomIndex = Math.floor(Math.random() * stims.length);
			stim = stims[randomIndex]
			stims.splice(randomIndex, 1);

			// enable and display the startButton
			let startButton = d3.select("#startButton");
			startButton.attr("disabled", null);
			startButton.style("display", "inline");
			
			// disable and hide the submitButton
			var submitButton = document.getElementById("submitButton");
			submitButton.removeEventListener("click", handleButtonClick); 
			submitButton.style.display = "none";
			submitButton.disabled = true;

			// shuffle positions of options
			let radioItems = Array.from(document.querySelectorAll('.radio-item'));
			radioItems.sort(() => Math.random() - 0.5);
			let container = document.querySelector('.radio-container');
			container.innerHTML = '';
			let column1 = document.createElement('div');
			column1.className = 'radio-column';
			let column2 = document.createElement('div');
			column2.className = 'radio-column';
			radioItems.forEach((item, index) => {
				if (index < 5) {
					column1.appendChild(item);
				} else {
					column2.appendChild(item);
				}
			});
				
			container.appendChild(column1);
			container.appendChild(column2);

			// disable the warning message
			var textContainer = document.getElementById("WARN_select");
			textContainer.innerHTML = "Please select one option before proceeding to the next page!";
			textContainer.style.fontSize = "24px";  
			textContainer.style.fontWeight = "bold"; 
			textContainer.style.color = "orange";  
			textContainer.style.display = "none";  
			
			// if startButton is clicked, show the video
			startButton.on("click", function() {
				show_video(stim["path"]);
			});	
		}
	};

	function handleButtonClick() {
		var checkedOption = document.querySelector('input[name="option"]:checked');
		if (checkedOption) {
			selectedOption = checkedOption.id 
			checkedOption.checked = false; // clear the option
				
			/* 
			Determine the modality type based on specific keywords present in the stimulus path.
			For example,

			if (stim["path"].includes("RGB")) {
				modality = "RGB";
			
			*/
			
			psiTurk.recordTrialData({
				'phase': "TEST",
				'video': stim["path"],
				'action': stim["label"],
				'response': selectedOption,
				'modality':modality
			});

			remove_video();
			next();

		}
		else{
			var textContainer = document.getElementById("WARN_select"); 
			textContainer.style.display = "inline";  
		}
		
	}

	var show_video = function(videoPath) {
		// when the video is showed, disable and hide the startButton
		let startButton = d3.select("#startButton");
		startButton.attr("disabled", true);
		startButton.style("display", "none");

		var video = d3.select("#stim")
		.append("video")
		.attr("id", "video")
		.attr("width", "398")
		.attr("height", "224")
		.attr("autoplay", true)
		.append("source")
		.attr("src", videoPath)
		.attr("type", "video/mp4");

		d3.select("#video").on("ended", function() {
			// when the video is end, enable and display the submitButton
			d3.select(this).style("display", "none");
			var submitButton = document.getElementById("submitButton");
			submitButton.style.display = "inline";
			submitButton.disabled = false;
			submitButton.addEventListener("click", handleButtonClick); 
		});			
		
	
	};

	var remove_video = function() {
		d3.select("#video").remove();
	};

	psiTurk.showPage('stage.html');

	// Start the test
	next();
};



/****************
* Questionnaire *
****************/

var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

	record_responses = function() {

		psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

		$('textarea').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});
		$('select').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);		
		});

	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);
		
		psiTurk.saveData({
			success: function() {
			    clearInterval(reprompt); 
                psiTurk.computeBonus('compute_bonus', function(){
                	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
                }); 
			}, 
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet 
	psiTurk.showPage('postquestionnaire.html');
	psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});
	
	$("#next").click(function () {
	    record_responses();
	    psiTurk.saveData({
            success: function(){
				psiTurk.completeHIT();
            }, 
            error: prompt_resubmit});
	});
    
	
};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
    psiTurk.doInstructions(
    	instructionPages, 
		function() { currentview = new VARExperiment_Instructions(); }, 
    );
});
