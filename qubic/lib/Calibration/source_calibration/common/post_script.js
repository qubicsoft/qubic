 const plot = document.getElementById('{plot_id}');

let lastClickTime = 0;
let lastTesId = null;
const doubleClickDelayMs = 350;

const overlay = document.createElement('div');

overlay.style.position = 'fixed';
overlay.style.top = '0';
overlay.style.left = '0';
overlay.style.width = '100vw';
overlay.style.height = '100vh';
overlay.style.background = 'rgba(0, 0, 0, 0.65)';
overlay.style.zIndex = '9999';
overlay.style.display = 'none';
overlay.style.alignItems = 'center';
overlay.style.justifyContent = 'center';

const panel = document.createElement('div');

panel.style.position = 'relative';
panel.style.width = '90vw';
panel.style.height = '90vh';
panel.style.background = 'white';
panel.style.borderRadius = '8px';
panel.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.35)';

const closeButton = document.createElement('button');

closeButton.textContent = 'Close';
closeButton.style.position = 'absolute';
closeButton.style.top = '10px';
closeButton.style.right = '10px';
closeButton.style.zIndex = '10001';
closeButton.style.padding = '6px 12px';
closeButton.style.cursor = 'pointer';

const detailPlot = document.createElement('div');

detailPlot.style.width = '100%';
detailPlot.style.height = '100%';

panel.appendChild(closeButton);
panel.appendChild(detailPlot);

overlay.appendChild(panel);

document.body.appendChild(overlay);

function closeDetailPlot() {
    overlay.style.display = 'none';
    Plotly.purge(detailPlot);
}

closeButton.addEventListener(
    'click',
    closeDetailPlot
);

overlay.addEventListener(
    'click',
    function(event) {
        if (event.target === overlay) {
            closeDetailPlot();
        }
    }
);

document.addEventListener(
    'keydown',
    function(event) {
        if (
            event.key === 'Escape'
            && overlay.style.display === 'flex'
        ) {
            closeDetailPlot();
        }
    }
);

plot.on(
    'plotly_click',
    function(eventData) {
        if (
            !eventData
            || !eventData.points
            || eventData.points.length === 0
        ) {
            return;
        }

        const point = eventData.points[0];
        const tesId = point.customdata;

        if (
            tesId === undefined
            || tesId === null
        ) {
            return;
        }

        const now = Date.now();

        const isDoubleClick = (
            tesId === lastTesId
            && now - lastClickTime <= doubleClickDelayMs
        );

        lastClickTime = now;
        lastTesId = tesId;

        if (!isDoubleClick) {
            return;
        }

        lastClickTime = 0;
        lastTesId = null;

        const trace = point.data;

        overlay.style.display = 'flex';

        Plotly.newPlot(
            detailPlot,
            [
                {
                    x: trace.x,
                    y: trace.y,
                    mode: 'lines',
                    type: 'scattergl',
                    name: `TES ${tesId}`,
                    line: trace.line,
                    opacity: trace.opacity ?? 1,
                    hovertemplate:
                        `TES ${tesId}<br>`
                        + 'time=%{x:.2f} s<br>'
                        + 'signal=%{y:.3g}'
                        + '<extra></extra>'
                }
            ],
            {
                title: `TES ${tesId} - TOD`,
                xaxis: {
                    title: 'Time [s]'
                },
                yaxis: {
                    title: 'Signal [ADU]'
                },
                margin: {
                    l: 70,
                    r: 30,
                    t: 70,
                    b: 60
                }
            },
            {
                responsive: true
            }
        );
    }
);
