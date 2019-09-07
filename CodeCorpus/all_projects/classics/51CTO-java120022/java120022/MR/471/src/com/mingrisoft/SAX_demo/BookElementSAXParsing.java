/**
 * @jdk版本:1.6
 * @编码时间:2010-3-20
 */
package com.mingrisoft.SAX_demo;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import com.mingrisoft.entity.BookElement;

/**
 * @author bwm
 * 
 */
public class BookElementSAXParsing extends DefaultHandler {

	private List<BookElement> bookList;

	private BookElement bookElement;

	private String elementValue;

	@Override
	public void startDocument() throws SAXException {
		// TODO Auto-generated method stub
		bookList = new ArrayList<BookElement>();
	}

	@Override
	public void startElement(String uri, String localName, String qName,
			Attributes attributes) throws SAXException {
		// TODO Auto-generated method stub

		// 每次读取book元素的时候重新初始化bookElement
		if (bookElement == null || "book".equals(localName)) {
			bookElement = new BookElement();
		}
	}

	/**
	 * 元素结束时，把元素内容置到book中
	 */
	@Override
	public void endElement(String uri, String localName, String qName)
			throws SAXException {
		// TODO Auto-generated method stub

		if ("name".equals(localName)) {
			bookElement.setName(elementValue);
		} else if ("publisher".equals(localName)) {
			bookElement.setPublisher(elementValue);
		} else if ("company".equals(localName)) {
			bookElement.setCompany(elementValue);
		} else if ("author".equals(localName)) {
			bookElement.setAuthor(elementValue);
		} else if ("ISBN".equals(localName)) {
			bookElement.setIsbn(elementValue);
		} else if ("price".equals(localName)) {
			bookElement.setPrice(Double.valueOf(elementValue));
		} else if ("url".equals(localName)) {
			try {
				bookElement.setUrl(new URL(elementValue));
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else if ("book".equals(localName)) {
			bookList.add(bookElement);
		}
	}

	/**
	 * 解析元素内容
	 */
	@Override
	public void characters(char[] ch, int start, int length)
			throws SAXException {
		// TODO Auto-generated method stub
		elementValue = elementValue.valueOf(ch, start, length);
		elementValue = elementValue.replace("\t", "");
		elementValue = elementValue.replace(" ", "");
		elementValue = elementValue.replace("\n", "");
		elementValue = elementValue.replace("\r", "");
	}

	public List<BookElement> getBookList() {
		return this.bookList;
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
		BookElementSAXParsing elementSAXParsing = new BookElementSAXParsing();
		elementSAXParsing.parseReadFile(pathname);
		System.out.println("自定义JAVA类封装元素名称和内容");
		System.out.println(elementSAXParsing.getBookList());
	}

}
