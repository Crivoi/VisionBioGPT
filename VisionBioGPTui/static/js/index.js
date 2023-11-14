(function init() {
    const btnText = {
        loading: 'Loading',
        ready: 'Send'
    };
    const submitBtn = document.getElementById('submit');
    const responseSection = document.getElementById('response');
    const form = document.getElementById('form');

    async function postPrompt(data) {
        const {csrfmiddlewaretoken, ...rest} = data;
        return await fetch('/api/model/', {
            method: 'POST',
            body: JSON.stringify(rest),
            headers: {
                'X-CSRFToken': csrfmiddlewaretoken,
                'Content-Type': 'application/json'
            }
        });
    }

    async function handleResponse(response) {
        console.log(response.status);
        if (response.ok) {
            const {text} = await response.json();
            responseSection.innerText = text;
        } else {
            alert(`Something went wrong!\nResponse code: ${response.status}`)
            const errorMessage = await response.text();
            console.error(`Error: ${response.status} - ${errorMessage}`);
        }
    }

    async function handleSubmit(e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        setBtnLoadingState();

        const response = await postPrompt(data);

        await handleResponse(response);

        setBtnReadyState();
    }

    function setBtnLoadingState() {
        submitBtn.setAttribute('disabled', 'disabled');
        submitBtn.innerText = btnText.loading;
    }

    function setBtnReadyState() {
        submitBtn.removeAttribute('disabled');
        submitBtn.innerText = btnText.ready;
    }

    form.addEventListener('submit', handleSubmit);
})();