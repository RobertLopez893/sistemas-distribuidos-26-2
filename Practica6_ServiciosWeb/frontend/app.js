document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('mlops-form');
    const statusConsole = document.getElementById('status-console');
    const statusText = document.getElementById('status-text');
    const btnEntrenar = document.getElementById('btn-entrenar');
    const leaderboardBody = document.getElementById('leaderboard-body');

    // Función para actualizar la tabla haciendo un GET a la API
    const actualizarLeaderboard = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/api/leaderboard');
            const data = await response.json();
            
            if (data.leaderboard.length > 0) {
                leaderboardBody.innerHTML = ''; // Limpiamos la tabla
                data.leaderboard.forEach((item, index) => {
                    const row = `
                        <tr>
                            <td><strong>#${index + 1}</strong></td>
                            <td>${item.modelo}</td>
                            <td><span class="badge bg-success">${item.accuracy}</span></td>
                            <td>${item.tiempo} s</td>
                        </tr>
                    `;
                    leaderboardBody.innerHTML += row;
                });
            }
        } catch (error) {
            console.error("Error al obtener el leaderboard", error);
        }
    };

    // Escuchamos el clic en "Desplegar Entrenamiento"
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const dataset = document.getElementById('dataset-select').value;
        const modelo = document.getElementById('model-select').value;

        statusConsole.classList.remove('d-none');
        statusConsole.classList.replace('alert-info', 'alert-warning');
        statusConsole.classList.replace('alert-success', 'alert-warning');
        statusText.innerHTML = `Orquestando <b>${modelo}</b>. Esperando respuesta de la API...`;
        btnEntrenar.disabled = true;

        const payload = {
            dataset: dataset,
            modelo: modelo
        };

        try {
            // El POST REAL hacia tu Backend en Python
            const response = await fetch('http://127.0.0.1:8000/api/entrenar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();

            if (data.status === "EXITO") {
                statusConsole.classList.replace('alert-warning', 'alert-success');
                statusText.innerHTML = `¡<b>${modelo}</b> procesado correctamente!`;
                // Actualizamos la tabla visualmente
                actualizarLeaderboard();
            }
            
        } catch (error) {
            statusConsole.classList.replace('alert-warning', 'alert-danger');
            statusText.innerHTML = `Error crítico de red. ¿Está encendido el servidor FastAPI?`;
        } finally {
            btnEntrenar.disabled = false;
        }
    });

    // Cargar el leaderboard al abrir la página por primera vez
    actualizarLeaderboard();
});
