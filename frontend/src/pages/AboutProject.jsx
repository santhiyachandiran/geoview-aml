import React, { useState } from 'react'

function Section({ icon, title, children }) {
  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
        <div style={{
          width: 40, height: 40, background: 'var(--brand-surface)',
          borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: '1.25rem', flexShrink: 0,
        }}>{icon}</div>
        <h2 style={{ fontSize: '1.125rem', fontWeight: 800, color: 'var(--text)' }}>{title}</h2>
      </div>
      <div className="prose">{children}</div>
    </div>
  )
}

export default function AboutProject() {
  return (
    <main className="page-body" style={{ maxWidth: 860 }}>
      <div className="section-header">
        <div>
          <h1 className="section-title">About the Project</h1>
          <p className="section-desc">
            What this application does, how it works, and what its limitations are.
          </p>
        </div>
        <span className="badge badge--info">Integrated M.Sc. AI &amp; ML Project</span>
      </div>

      <Section icon="🌍" title="The real-world problem">
        <p>
          Cities grow. Farmland shrinks. Forests are cleared. Understanding <strong>where
          and how fast</strong> these changes are happening is important for urban planning,
          environmental monitoring, and policy decisions.
        </p>
        <p>
          Satellite imagery lets us observe land from above at regular intervals. By comparing
          images taken years apart, we can detect which areas have changed from green vegetation
          or open fields to concrete, roads, and buildings.
        </p>
        <p>
          This project focuses on <strong>Coimbatore, Tamil Nadu, India</strong>, one of the
          fastest-growing cities in South India, where the IT corridor and urban expansion
          have significantly altered the land around the city since 2017.
        </p>
      </Section>

      <Section icon="🛰" title="Satellite imagery — Sentinel-2">
        <p>
          We use <strong>Sentinel-2</strong> satellite imagery from the European Space Agency.
          Sentinel-2 takes images of the same area roughly every 5 days, at a spatial resolution
          of <strong>10 metres per pixel</strong> for the bands we use.
        </p>
        <p>
          Because cloud cover is high over Tamil Nadu from June to October, we select images
          from the <strong>dry season (November to February)</strong> each year. This gives us
          comparable images year on year without cloud interference distorting the results.
        </p>
        <p>
          Multiple images from the same season are combined into a <strong>median composite</strong>:
          for each pixel, we take the middle value across all cloud-free observations.
          This further reduces the effect of remaining clouds and data gaps.
        </p>
        <details className="tech-details" style={{ marginTop: '1rem' }}>
          <summary className="tech-details__summary">Technical: Sentinel-2 bands used</summary>
          <div className="tech-details__body">
            <ul>
              <li><strong>B2</strong> – Blue (490 nm, 10 m)</li>
              <li><strong>B3</strong> – Green (560 nm, 10 m)</li>
              <li><strong>B4</strong> – Red (665 nm, 10 m)</li>
              <li><strong>B8</strong> – Near-infrared / NIR (842 nm, 10 m)</li>
              <li><strong>B11</strong> – Short-wave infrared / SWIR1 (1610 nm, 20 m)</li>
              <li><strong>B12</strong> – Short-wave infrared / SWIR2 (2190 nm, 20 m)</li>
              <li><strong>SCL</strong> – Scene Classification Layer (used for cloud masking)</li>
            </ul>
            <p>Collection: <code>COPERNICUS/S2_SR_HARMONIZED</code> on Google Earth Engine.</p>
          </div>
        </details>
      </Section>

      <Section icon="📊" title="Spectral signals — what do the numbers mean?">
        <p>
          Satellites do not directly tell us what is on the ground. Instead, they measure
          how much light is reflected back from the surface at different wavelengths.
          We then calculate <strong>spectral indices</strong> — ratios of these reflectance
          values — that correlate with specific land surface properties.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', margin: '1rem 0' }}>
          {[
            { icon: '🌿', title: 'Vegetation Greenness', desc: 'Healthy green vegetation reflects strongly in the near-infrared band and absorbs red light. Higher values mean more green vegetation cover.', note: 'Also called NDVI' },
            { icon: '💧', title: 'Water-Related Signal', desc: 'Water bodies and wet soil reflect green light but absorb infrared. A higher value indicates wetter surfaces or open water.', note: 'Also called NDWI' },
            { icon: '🏗️', title: 'Built-up Surface Signal', desc: 'Paved surfaces and buildings reflect more short-wave infrared than near-infrared. A higher value is consistent with built-up land.', note: 'Also called NDBI' },
          ].map(s => (
            <div key={s.title} className="card card--compact" style={{ background: 'var(--surface-2)' }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{s.icon}</div>
              <div style={{ fontWeight: 700, marginBottom: '0.4rem' }}>{s.title}</div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{s.desc}</div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>{s.note}</div>
            </div>
          ))}
        </div>
        <div className="alert alert--info" style={{ marginTop: '0.75rem' }}>
          <span className="alert__icon">ℹ️</span>
          <div>
            <div className="alert__title">Important: signals are proxies, not measurements</div>
            A high built-up signal can also appear in dry bare soil or rocky areas. A drop in
            vegetation greenness could mean a dry year, seasonal crop harvest, or land clearing —
            they are not automatically distinguishable without additional validation data.
          </div>
        </div>
      </Section>

      <Section icon="🤖" title="Machine learning — Random Forest and XGBoost">
        <p>
          Once real satellite data and reference land-cover labels are available, the application
          trains <strong>supervised classification models</strong> that learn to identify land-cover
          types from the combination of spectral band values and indices.
        </p>
        <p>
          <strong>Random Forest</strong> is the baseline: an ensemble of many decision trees that
          together vote on the most likely land-cover class for each pixel. It is fast and
          interpretable.
        </p>
        <p>
          <strong>XGBoost</strong> is the advanced model: a gradient-boosted tree ensemble whose
          hyperparameters are tuned automatically using Optuna. It typically achieves better
          performance on imbalanced multi-class problems.
        </p>
        <p>
          Both models are trained on <strong>ESA WorldCover 2021</strong> labels,
          using a <strong>spatial block split</strong> — meaning pixels from different
          geographic tiles are held out for testing, rather than a random split.
          This avoids inflating accuracy by training on pixels adjacent to the test set.
        </p>
        <div className="alert alert--demo" style={{ marginTop: '0.75rem' }}>
          <span className="alert__icon">📋</span>
          <div>
            <div className="alert__title">WorldCover temporal limitation</div>
            ESA WorldCover labels are from 2021. Using them to train a model applied to 2018 or 2024
            data introduces <strong>temporal label mismatch</strong>. This is documented as a known
            limitation and is standard practice in remote sensing research, but it must be noted
            in any published results.
          </div>
        </div>
      </Section>

      <Section icon="⚠️" title="Limitations and honest caveats">
        <ul>
          <li><strong>Demo mode:</strong> When Google Earth Engine is not authenticated, all numerical
            results are synthetic illustrations. They must not be cited as real satellite measurements.</li>
          <li><strong>Cloud cover:</strong> Despite filtering, residual thin cloud, haze, or cloud shadows
            can affect reflectance values and create apparent change where none occurred.</li>
          <li><strong>Seasonal mismatch:</strong> Even within the dry season, different year composites
            may have different effective dates, rainfall, or phenological stage.</li>
          <li><strong>Resolution:</strong> At 10 m per pixel, narrow roads, small water bodies, and
            buildings smaller than one pixel are mixed with surrounding land cover.</li>
          <li><strong>Classification uncertainty:</strong> Spectral overlap between bare soil and
            built-up surfaces, and between irrigated crops and forest, leads to misclassification.
            Confidence scores help quantify this but do not eliminate it.</li>
          <li><strong>Label year:</strong> ESA WorldCover 2021 labels do not describe 2018 or 2024
            conditions. Change detected by comparing classified maps may partly reflect label
            inaccuracy rather than real change.</li>
        </ul>
      </Section>

      <Section icon="📚" title="References and data sources">
        <ul>
          <li>
            <strong>Sentinel-2 Imagery:</strong> Copernicus Sentinel-2 Level-2A Surface Reflectance
            Harmonized. European Space Agency, distributed via Google Earth Engine.
            Collection: <code>COPERNICUS/S2_SR_HARMONIZED</code>.
          </li>
          <li>
            <strong>Land-cover labels:</strong> Zanaga et al. (2022). ESA WorldCover 10 m 2021 v200.
            Zenodo. <a href="https://doi.org/10.5281/zenodo.7254221" target="_blank" rel="noreferrer" style={{ color: 'var(--water)' }}>https://doi.org/10.5281/zenodo.7254221</a>.
            Licensed CC BY 4.0.
          </li>
          <li>
            <strong>Base map:</strong> © OpenStreetMap contributors. Licensed ODbL.
          </li>
          <li>
            <strong>GEE platform:</strong> Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S.,
            Thau, D., &amp; Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial
            analysis for everyone. <em>Remote Sensing of Environment</em>, 202, 18–27.
          </li>
        </ul>
      </Section>
    </main>
  )
}
