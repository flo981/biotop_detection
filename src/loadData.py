import geopandas as gpd
import folium
from folium.folium import Map
from pyproj import Transformer
from selenium import webdriver
import os
import io
from PIL import Image
from folium.features import DivIcon


def getText(temp_biotop):
    temp_nummer = temp_biotop['Nummer'][temp_biotop['Nummer'].keys()[0]]
    temp_gemnr = temp_biotop['gemnr'][temp_biotop['gemnr'].keys()[0]]
    temp_biotoptyp = temp_biotop['biotoptyp'][temp_biotop['biotoptyp'].keys()[0]]
    temp_bezeich = temp_biotop['bezeich'][temp_biotop['bezeich'].keys()[0]]
    text = "Nummer: " + str(temp_nummer) + "\n" + "GemNr: " + str(temp_gemnr) + "\n" + "Biotoptyp: " + str(
        temp_biotoptyp) + "\n" + "Bezeichnung: " + str(temp_bezeich)
    return text

def style_fcn(x):
    return {'stroke':True,'color': '#FF0000', 'fillColor': '#AARRGGBB','opacity':1, 'weight':2, 'line_cap':'round', 'fill':False}


def style_fcn_hull(x):
    return {'stroke':True,'color': '#008FFF', 'fillColor': '#AARRGGBB','opacity':1, 'weight':2, 'line_cap':'round', 'fill':False}


def style_fcn_env(x):
    return {'stroke':True,'color': '#F700FF', 'fillColor': '#AARRGGBB','opacity':1, 'weight':2, 'line_cap':'round', 'fill':False}



def biotop_center(temp_biotop):
    #get biotop pos
    center_temp = temp_biotop.geometry.centroid
    key = center_temp.keys()[0]
    return transformer.transform(center_temp[key].x,center_temp[key].y)

def biotop_current_map(temp_location, text):
    #TODO: Linie weiter ausen umrunden lassen oder zusätzliche box
    m_temp = Map(tiles=None, location=[temp_location[1], temp_location[0]],
                 zoom_start=18,
                 prefer_canvas=True,
                 no_touch=True,
                 disable_3d=True,
                 attr="test",
                 control_scale=True,
                 zoom_control=False)

    atlas = folium.raster_layers.WmsTileLayer(url=wmts, layers="bmaporthofoto30cm",
                                              fmt='image/png',
                                              transparent=True,
                                              overlay=True).add_to(m_temp)

    # temp_biotop_scaled = temp_biotop.scale(xfact=2, yfact=2, zfact=0, origin='center')
    temp_biotop_convex_hull = temp_biotop.convex_hull
    temp_biotop_envelope = temp_biotop.envelope

    folium.GeoJson(temp_biotop, style_function=style_fcn).add_to(m_temp)
    # folium.GeoJson(temp_biotop_scaled,style_function=style_fcn).add_to(m_temp)
    #folium.GeoJson(temp_biotop_convex_hull, style_function=style_fcn_hull).add_to(m_temp)
    #folium.GeoJson(temp_biotop_envelope, style_function=style_fcn_env).add_to(m_temp)

    #folium.map.Marker([temp_location[1] + 0.001, temp_location[0] - 0.001],
    #                icon=DivIcon(
    #                    icon_size=(150, 36),
    #                    icon_anchor=(0, 0),
    #                    html='<div style="font-size: 10pt;"><span style="color: #ff0000;">%s</span></div>' % text)).add_to(m_temp)
    return m_temp

def save_current_biotop(m_temp, bio_number):
    png = m_temp._to_png()
    out = open(os.path.join(script_dir, '../data/output_biotop/'+bio_number+'.png'), 'wb')
    bytes_written = out.write(png)
    out.close()

def save_current_biotop2(m_temp, bio_number,case):
    img_data = m_temp._to_png()
    img = Image.open(io.BytesIO(img_data))
    img.save(os.path.join(script_dir, '../data/output_biotop/'+str(case)+'_'+bio_number+'.png'))

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    nReserve = gpd.read_file(os.path.join(script_dir, '../data/Biotopkartierung/Biotopkartierung.shp'))

    biotop_1 = nReserve.query("biotoptyp=='8.1.1.1'")
    biotop_2 = nReserve.query("biotoptyp=='8.1.1.2'")

    wmts = "http://maps.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg"
    #driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver')
    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')

    transformer = Transformer.from_crs(31258, 4326, always_xy=True)  # 1frpm 2to


    len_bio_1 = biotop_1.shape[0]
    len_bio_2 = biotop_2.shape[0]

    biotop_key_nummer = biotop_1['Nummer']
    biotop_keys = biotop_key_nummer.keys()
    for i in range(0, len_bio_1-1):
        key_i = biotop_keys[i]
        bio_i = biotop_1['Nummer'][key_i]
        str_nummer = "\'" + bio_i + "\'"
        query_str = "Nummer==" + str(str_nummer)
        temp_biotop = biotop_1.query(query_str)
        print("Process Biotop (1): ", i, " ", key_i)
        temp_location = biotop_center(temp_biotop)
        m_temp = biotop_current_map(temp_location,getText(temp_biotop))
        save_current_biotop2(m_temp, bio_i, 1)

    biotop_key_nummer = biotop_2['Nummer']
    biotop_keys = biotop_key_nummer.keys()
    for i in range(0, len_bio_2-1):
        key_i = biotop_keys[i]
        bio_i = biotop_2['Nummer'][key_i]
        str_nummer = "\'" + bio_i + "\'"
        query_str = "Nummer==" + str(str_nummer)
        temp_biotop = biotop_2.query(query_str)
        print("Process Biotop (2): ", i, " ", key_i)
        temp_location = biotop_center(temp_biotop)
        m_temp = biotop_current_map(temp_location,str(bio_i))
        save_current_biotop2(m_temp, bio_i, 2)


#TODO: down-size images
