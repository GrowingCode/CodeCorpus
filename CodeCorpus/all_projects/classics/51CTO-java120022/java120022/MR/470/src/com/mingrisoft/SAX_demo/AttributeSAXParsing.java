/**
 * @jdk版本:1.6
 * @编码时间:2010-3-20
 */
package com.mingrisoft.SAX_demo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * @author bwm
 * 
 */
public class AttributeSAXParsing extends DefaultHandler {

	List<String> attribute = new ArrayList<String>();

	/**
	 * 读取属性名称和属性值保存在List中
	 */
	@Override
	public void startElement(String uri, String localName, String qName,
			Attributes attributes) throws SAXException {
		// 读取属性名称和属性值保存在List中
		for (int i = 0; i < attributes.getLength(); i++) {
			attribute.add(localName + " = " + attributes.getLocalName(i) + ":"
					+ attributes.getValue(i));
		}
	}

	public List<String> getAttribute() {
		return this.attribute;
	}

	/**
	 * 通过文件读取XML
	 * 
	 * @param pathname
	 *            文件路径
	 */
	public void parseReadFile(String pathname) {
		SAXParser parser;
		SAXParserFactory factory = SAXParserFactory.newInstance();
		try {
			factory.setValidating(true);
			factory.setNamespaceAware(true);
			parser = factory.newSAXParser();
			File file = new File(pathname);
			parser.parse(file, this);
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static void main(String[] arg) {

		String pathname = "xmldemo/books.xml";
		AttributeSAXParsing elementSAXParsing = new AttributeSAXParsing();
		elementSAXParsing.parseReadFile(pathname);
		System.out.println("属性名称和属性值");
		System.out.println(elementSAXParsing.getAttribute());
	}
}
